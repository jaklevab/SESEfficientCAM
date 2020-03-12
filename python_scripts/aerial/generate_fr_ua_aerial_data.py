import geopandas as gpd
from geopandas import GeoDataFrame
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
warnings.filterwarnings("ignore")
from shapely.geometry import MultiLineString
from shapely.ops import polygonize
import json
from fiona.crs import from_epsg
from rasterio.mask import mask
from joblib import Parallel, delayed
import skimage
from skimage import io
import json
import sys
import glob

# Global paths
BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
sys.path.append(BASE_DIR + "SATELSES/equirect_proj_test/cnes/python_files/")
from generate_fr_ua_vhr_data import *
AERIAL_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/ign/ORTHO_HR/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/AERIAL_esa_URBAN_ATLAS_FR/"
MAX_NB_JOBS = 40
INTER_OUT_DIR = OUTPUT_DIR + "inter_OUTPUTS/"

def generate_aerial_data():
    # AERIAL DATA
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
    if ori_tile.endswith('.jp2'):
        reproj_tile = ori_tile.replace('.jp2','_rprj_3035.tif')
    else:
        reproj_tile = ori_tile.replace('.tif','_rprj_3035.tif')
    os.system("~/anaconda3/envs/bv_env/bin/gdalwarp -t_srs epsg:3035 -r bilinear -co BIGTIFF=YES -co COMPRESS=LZW %s %s"%(ori_tile,reproj_tile))
    print("Projected file: %s"%reproj_tile)
    return reproj_tile

def merge_coll_tiles(ori_tiles):
    src_files_to_mosaic = [rasterio.open(x) for x in ori_tiles.split(";")]
    file_comb_outname = "_".join(["-".join(x.split("/")[-1].split(".")[0].split("-")[:4]) for x in ori_tiles.split(";")])
    inter_prefix = '' #"_".join([x.split("/")[10] for x in ori_tiles.split(";")])
    merged_tile = INTER_OUT_DIR + inter_prefix + file_comb_outname + "_merged.tif"
    #print("%d files to merge"%len(src_files_to_mosaic))
    mosaic, out_trans = merge(src_files_to_mosaic)
    print("Finished Merging")
    out_meta = src_files_to_mosaic[0].meta.copy()
    #print(src_files_to_mosaic[0].crs)
    out_crs = src_files_to_mosaic[0].crs #["init"]
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": out_crs,
                    })
    print("Writing merged to output")
    with rasterio.open(merged_tile, "w", **out_meta) as dest:
        dest.write(mosaic)
    print("Close all files")
    for src in src_files_to_mosaic:
        src.close()
    print("Merged file: %s"%merged_tile)
    return merged_tile

def subextract_from_aerial_tile(tile_fnames,insee_cells_to_extract):
    print(tile_fnames)
    print(insee_cells_to_extract)
    if len(tile_fnames.split(";")) ==1 :
        print("Unique Tiles")
        #No need to merge
        reproj_tile = reproject_tile(tile_fnames)
        aerial_data = rasterio.open(reproj_tile)
        curr_sat_dir = tile_fnames.split("/")[10]
    else:
        print("Multiple Tiles")
        merged_tile = merge_coll_tiles(tile_fnames)
        reproj_tile = reproject_tile(merged_tile)
        aerial_data = rasterio.open(reproj_tile)
        curr_sat_dir = "_".join(list(set([x.split("/")[10] for x in tile_fnames.split(";")])))
    out_dir = OUTPUT_DIR + curr_sat_dir
    if not os.path.exists(out_dir):
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass
    # EXTRACT
    for it in range(insee_cells_to_extract.shape[0]):
        geo = insee_cells_to_extract.iloc[it:(it+1)]
        coords = getFeatures(geo)
        idINSPIRE, _ = geo.values[0]
        out_img, out_transform = rasterio.mask.mask(dataset=aerial_data, shapes=coords, crop=True)
        reorder_img = np.dstack([out_img[0],out_img[1],out_img[2]])
        io.imsave(out_dir+"/FR_URBANATLAS_200m_%s.png"%idINSPIRE,skimage.img_as_ubyte(reorder_img))
    aerial_data.close()
    os.remove(reproj_tile)
    if len(tile_fnames.split(";")) > 1 :
        os.remove(merged_tile)
    print("Done")
    return None

def extract_labelled_aerial_imagery(df_fullmerge2insee):
    nb_tiles = df_fullmerge2insee.shape[0]
    n_jbs = min(nb_tiles, MAX_NB_JOBS)
    prepare_input = [(aerial_fname,
                      gpd.GeoDataFrame(pd.DataFrame([idINSPIRE,insee_geom]).transpose().rename(
                          columns={0:'idINSPIRE',1:'geometry'}),crs={"init":"epsg:3035"}))
                     for aerial_fname, idINSPIRE, insee_geom in tqdm(df_fullmerge2insee.values)]
    test_prepare_input = prepare_input
    print("Treating %d out of %d aerial tiles"%(len(test_prepare_input),len(prepare_input)))
    full_data =  Parallel(n_jobs=n_jbs)(
        delayed(subextract_from_aerial_tile)
        (tile_fnames=aerial_fname,insee_cells_to_extract=gdf_to_extract) 
        for aerial_fname,gdf_to_extract in tqdm(test_prepare_input))
    #full_data = [subextract_from_aerial_tile(aerial_fname,gdf_to_extract) for aerial_fname,gdf_to_extract in tqdm(prepare_input)]

def main():
    #
    print("Generating Census")
    geo_df_full_census_data = generate_complete_census_data()
    geo_df_full_census_data = geo_df_full_census_data.to_crs({"init":"epsg:3035"})
    already_extracted_idINSPIRE = pd.DataFrame([g.split("_")[-1].split(".")[0]
                                   for f in os.listdir(OUTPUT_DIR) for g in os.listdir(OUTPUT_DIR + f) if g.endswith(".png")],
                                   columns=["idINSPIRE"])
    geo_df_full_census_data = pd.merge(geo_df_full_census_data,already_extracted_idINSPIRE,how='outer', indicator=True)
    geo_df_full_census_data = geo_df_full_census_data[geo_df_full_census_data._merge!="both"]
    geo_df_full_census_data.drop(["_merge"],axis=1,inplace=True)
    
    #
    print("Generating UA Boundaries")
    geo_ua_bound = generate_urban_atlas_boundaries()
    geo_census_ua = gpd.sjoin(geo_df_full_census_data,geo_ua_bound).drop("index_right",axis=1)
    #
    print("Generating Aerial Data")
    geo_aerial_data = generate_aerial_data()
    geo_aerial_data = geo_aerial_data.to_crs({"init":"epsg:3035"})
    df_insee_cars_2_aerial = gpd.sjoin(geo_census_ua,geo_aerial_data,op="intersects")
    #
    extraction_df = df_insee_cars_2_aerial.groupby(["idINSPIRE"])["NOM"].apply(lambda x : list(x)).reset_index()
    extraction_df["aerial_par"] = [";".join(set(x)) for x in extraction_df.NOM]
    extraction_df = pd.merge(extraction_df,geo_df_full_census_data[["idINSPIRE","geometry"]],on="idINSPIRE")
    #
    merge2insee_id = extraction_df.groupby("aerial_par")["idINSPIRE"].apply(lambda x : list(x)).reset_index()
    merge2insee_geometries = extraction_df.groupby("aerial_par")["geometry"].apply(lambda x : list(x)).reset_index()
    merge2insee = pd.merge(merge2insee_id,merge2insee_geometries,on="aerial_par")
    print("Proceeding extract images")
    extract_labelled_aerial_imagery(merge2insee)

if __name__ == "__main__":
    main()



