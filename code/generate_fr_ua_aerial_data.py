import geopandas as gpd
from geopandas import GeoDataFrame as gdf
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import rasterio
import os, re
from rasterio.merge import merge
from tqdm import tqdm
from rasterio import plot
from shapely.geometry import MultiLineString
from shapely.ops import polygonize
from fiona.crs import from_epsg
from rasterio.mask import mask
from joblib import Parallel, delayed
import skimage
from skimage import io
import json
import sys
import glob
import multiprocessing

# Global paths
DATA_BASE_DIR = "../data/"
OUTPUT_BASE_DIR = "../results/"
AERIAL_DIR = DATA_BASE_DIR + "aerial_data/"
CENSUS_DIR = DATA_BASE_DIR + 'census_data/'
UA_DIR = DATA_BASE_DIR + "UA_data/"
OUTPUT_DIR = OUTPUT_BASE_DIR + "imagery_out/"
INTER_OUT_DIR = OUTPUT_DIR + "inter_OUTPUTS/"
MAX_NB_JOBS = 1 #min(multiprocessing.cpu_count(),40) # Change to speed up extraction in case multicore architecture is available

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame
    in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def generate_urban_atlas_boundaries():
    ''' Generates Boundaries for each city in the EU Urban Atlas'''
    ua_bounds = [
        (ua.split("_")[-1],
         gpd.read_file(UA_DIR + ua +"/Shapefiles/" + x).geometry.values[0])
        for ua in os.listdir(UA_DIR) for x in os.listdir(UA_DIR + ua + "/Shapefiles/")
        if x.endswith("_UA2012_Boundary.shp")
    ]
    ua_gdf = gpd.GeoDataFrame(ua_bounds,columns=["city","geometry"])
    ua_gdf.crs = {"init":"epsg:3035"}
    return ua_gdf

def generate_aerial_data():
    """ Returns geographical boundaries of each tile in the image dataset"""
    shapes_dir = glob.glob(AERIAL_DIR + "**/**/ORTHOHR/3*/**/*dalles.shp")
    full = []

    for d in shapes_dir:
        test = gpd.read_file(d)
        base_dir =  glob.glob("/".join(d.split("/")[:-1])+"/../../1*/*/")[0]
        test["NOM"] = [base_dir + x for x in test.NOM]
        full.append(test)

    full_dalles = gpd.GeoDataFrame(pd.concat(full))
    full_dalles.crs = full[0].crs
    return full_dalles

def reproject_tile(ori_tile):
    """ Reproject tiles into {epsg:3035}"""

    # Change the name of the reprojected tile
    if ori_tile.endswith('.jp2'):
        reproj_tile = ori_tile.replace('.jp2','_rprj_3035.tif')
    else:
        reproj_tile = ori_tile.replace('.tif','_rprj_3035.tif')

    # Reproject tile into epsg:3035 CRS
    os.system(
        "gdalwarp -t_srs epsg:3035 -r bilinear -co BIGTIFF=YES -co COMPRESS=LZW %s %s"\
        %(ori_tile,reproj_tile))
    return reproj_tile

def merge_coll_tiles(ori_tiles):
    """ Merge together adjacent tiles to extract census cells images
    intersecting both tiles"""

    # Merge rasters
    src_files_to_mosaic = [rasterio.open(x) for x in ori_tiles.split(";")]
    file_comb_outname = "_".join(["-".join(x.split("/")[-1].split(".")[0].split("-")[:4])
                                  for x in ori_tiles.split(";")])
    merged_tile = INTER_OUT_DIR  + file_comb_outname + "_merged.tif"
    mosaic, out_trans = merge(src_files_to_mosaic)

    # Define the merged raster crs
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_crs = src_files_to_mosaic[0].crs
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": out_crs})

    # Output and close
    with rasterio.open(merged_tile, "w", **out_meta) as dest:
        dest.write(mosaic)
    for src in src_files_to_mosaic:
        src.close()
    return merged_tile

def subextract_from_aerial_tile(tile_fnames,insee_cells_to_extract):
    """ Extract all census cell images contained in a given aerial tile"""

    if len(tile_fnames.split(";")) ==1 :

        # Census cells images contained within a single aerial tile (no overlap)
        reproj_tile = reproject_tile(tile_fnames)
        aerial_data = rasterio.open(reproj_tile)
        curr_sat_dir = tile_fnames.split("/")[10]

    else:

        # Census cells images contained within a overlapping aerial tiles (merge needed)
        merged_tile = merge_coll_tiles(tile_fnames)
        reproj_tile = reproject_tile(merged_tile)
        aerial_data = rasterio.open(reproj_tile)
        curr_sat_dir = "_".join(list(set([x.split("/")[10]
                                          for x in tile_fnames.split(";")])))

    # Generate output directory for images extracted from overlapping tiles
    out_dir = OUTPUT_DIR + curr_sat_dir
    if not os.path.exists(out_dir):
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass

    # Extract images from tile and save them
    for it in range(insee_cells_to_extract.shape[0]):
        geo = insee_cells_to_extract.iloc[it:(it+1)]
        coords = getFeatures(geo)
        idINSPIRE, _ = geo.values[0]
        out_img, out_transform = rasterio.mask.mask(dataset=aerial_data,
                                                    shapes=coords, crop=True)
        reorder_img = np.dstack([out_img[0],out_img[1],out_img[2]])
        io.imsave(out_dir+"/FR_URBANATLAS_200m_%s.png"%idINSPIRE,
                  skimage.img_as_ubyte(reorder_img))
    aerial_data.close()

    # Remove reprojected and merged tiles
    os.remove(reproj_tile)
    if len(tile_fnames.split(";")) > 1 :
        os.remove(merged_tile)
    return None

def extract_labelled_aerial_imagery(df_fullmerge2insee):
    nb_tiles = df_fullmerge2insee.shape[0]
    n_jbs = min(nb_tiles, MAX_NB_JOBS)

    # Prepare input to extract images
    prepare_input = [
        (aerial_fname,
         gpd.GeoDataFrame(pd.DataFrame([idINSPIRE,insee_geom]).transpose().rename(
                          columns={0:'idINSPIRE',1:'geometry'}),
                          crs={"init":"epsg:3035"}))
        for aerial_fname, idINSPIRE, insee_geom in tqdm(df_fullmerge2insee.values)]

    # Extract images
    if MAX_NB_JOBS > 1:
        full_data =  Parallel(n_jobs=n_jbs)(
            delayed(subextract_from_aerial_tile)(aerial_fname,gdf_to_extract)
            for aerial_fname,gdf_to_extract in tqdm(prepare_input))
    else:
        full_data =  [subextract_from_aerial_tile(aerial_fname,gdf_to_extract)
                      for aerial_fname,gdf_to_extract in tqdm(prepare_input)]

def main():

    # Generate Census Data as provided by INSEE
    geo_df_data = gdf.from_file(CENSUS_DIR+ 'Filosofi2015_carreaux_200m_metropole.shp')
    geo_df_data.rename({"IdINSPIRE":"idINSPIRE"},axis=1,inplace=True)
    geo_df_data["income"] = geo_df_data.Ind_snv/geo_df_data.Ind
    geo_df_data[["idINSPIRE","income"]].to_csv(
        CENSUS_DIR + "squares_to_ses_2019.csv",index=False)
    geo_df_data = geo_df_data.to_crs({"init":"epsg:3035"})

    # Extract only images that haven't yet been extracted
    already_extracted_idINSPIRE = pd.DataFrame([g.split("_")[-1].split(".")[0]
                                   for f in os.listdir(OUTPUT_DIR)
                                   for g in os.listdir(OUTPUT_DIR + f)
                                   if g.endswith(".png")],columns=["idINSPIRE"])
    geo_df_data = pd.merge(geo_df_data,already_extracted_idINSPIRE,
                           how='outer', indicator=True)
    geo_df_data = geo_df_data[geo_df_data._merge!="both"]
    geo_df_data.drop(["_merge"],axis=1,inplace=True)

    # Generate Urban Atlas city boundaries to associate each census cell to a city
    geo_ua_bound = generate_urban_atlas_boundaries()
    geo_census_ua = gpd.sjoin(geo_df_data,geo_ua_bound).drop("index_right",axis=1)
    extract_ua = geo_census_ua[["idINSPIRE","city"]]
    extract_ua["FUA_NAME"] = [x[0:1].upper() + x[1:].lower() for x in extract_ua.city]
    extract_ua[["idINSPIRE","FUA_NAME"]].to_csv(AERIAL_DIR + "city_assoc.csv",index=False)

    # Define Aerial Data tiles as provided by IGN
    geo_aerial_data = generate_aerial_data()
    geo_aerial_data = geo_aerial_data.to_crs({"init":"epsg:3035"})

    # Determine all census cells included in a given aerial tile
    df_insee2aerial = gpd.sjoin(geo_census_ua,geo_aerial_data,op="intersects")
    extraction_df = df_insee2aerial.groupby(["idINSPIRE"])["NOM"]\
                                   .apply(lambda x : list(x)).reset_index()
    extraction_df["aerial_par"] = [";".join(set(x)) for x in extraction_df.NOM]
    extraction_df = pd.merge(extraction_df,geo_df_data[["idINSPIRE","geometry"]],
                             on="idINSPIRE")
    merge2insee_id = extraction_df.groupby("aerial_par")["idINSPIRE"]\
                                  .apply(lambda x : list(x)).reset_index()
    merge2insee_geom = extraction_df.groupby("aerial_par")["geometry"]\
                                  .apply(lambda x : list(x)).reset_index()
    merge2insee = pd.merge(merge2insee_id,merge2insee_geom,on="aerial_par")

    # Extract all census cell images from each tile
    extract_labelled_aerial_imagery(merge2insee)

    # List all extracted images and associate to their census ID
    image_files = [os.path.join(inter_sat_dir,im_file)
                   for inter_sat_dir in (os.listdir(OUTPUT_DIR))
                   if not inter_sat_dir.endswith(".csv")
                   for im_file in os.listdir(OUTPUT_DIR + inter_sat_dir)
                    if im_file.endswith(".png")]
    #
    im_df = pd.DataFrame()
    im_df["path2im"] = image_files
    im_df["idINSPIRE"] = [k.split("/")[-1].split(".")[0].split("_")[-1]
                          for k in image_files]

    # Exclude any images containing empty pixels
    check_void = parallel_make_dataset(im_df.path2im,CPU_USE=MAX_NB_JOBS)
    void_df = pd.DataFrame(check_void,columns=["path2im","non_void"])
    full_im_df_all = pd.merge(im_df,void_df,on="path2im")
    full_im_df = full_im_df_all[full_im_df_all.non_void]
    full_im_df.reset_index(drop=True,inplace=True)
    full_im_df_all[["idINSPIRE","non_void"]]\
                    .to_csv(AERIAL_DIR + "void_data.csv",index=False)

if __name__ == "__main__":
    main()
