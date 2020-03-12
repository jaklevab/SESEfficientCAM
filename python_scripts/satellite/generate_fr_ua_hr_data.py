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
from generate_fr_ua_vhr_data import *

BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/500m_CONTEXT_esa_URBAN_ATLAS_FR/"
MAX_NB_JOBS = 10
INSEE_SIZE = 200
WINDOW_SIZE = 500

def subextract_HR_from_tile(sat_fname,gdf_to_extract,window_size = WINDOW_SIZE):
    sat_data = rasterio.open(sat_fname)
    curr_sat_dir = sat_fname.split("/")[-2]
    if not os.path.exists(OUTPUT_DIR + curr_sat_dir):
        os.makedirs(OUTPUT_DIR + curr_sat_dir)
    #
    for it in range(gdf_to_extract.shape[0]):
        geo = gdf_to_extract.iloc[it:(it+1)].to_crs(sat_data.crs)
        geo_data = pd.DataFrame(list(geo.geometry.iloc[0].exterior.coords),columns=["x","y"])
        left,right,bottom,up = min(geo_data.x),max(geo_data.x),min(geo_data.y),max(geo_data.y)        
        size_to_pad = (window_size - INSEE_SIZE)/2
        new_coords = (
            (left-size_to_pad,bottom-size_to_pad),
            (right+size_to_pad,bottom-size_to_pad),
            (right+size_to_pad,up+size_to_pad),
            (left-size_to_pad,up+size_to_pad),
            (left-size_to_pad,bottom-size_to_pad),
        )
        new_geo = gpd.GeoDataFrame([Polygon(new_coords),],columns=["geometry"])
        new_coords = getFeatures(new_geo)
        _, idINSPIRE = geo.values[0]
        out_img, out_transform = rasterio.mask.mask(dataset=sat_data, shapes=new_coords, crop=True)
        reorder_img = np.dstack([out_img[0],out_img[1],out_img[2]])
        io.imsave(OUTPUT_DIR + curr_sat_dir + "/FR_URBANATLAS_%dm_%s.png"%(window_size,idINSPIRE),skimage.img_as_ubyte(reorder_img))
    sat_data.close()
    print("Done")
    return None

def extract_labelled_imagery(sat_square_to_tiff):
    tiff_to_squares = sat_square_to_tiff.groupby("sat_data")[["geometry","idINSPIRE"]].apply(lambda x : list(x.values)).reset_index()
    nb_tiles = tiff_to_squares.shape[0]
    n_jbs = min(nb_tiles, MAX_NB_JOBS)
    prepare_input = [(sat_fname,gpd.GeoDataFrame(pd.DataFrame(values_l,columns=["geometry","idINSPIRE"]),crs=sat_square_to_tiff.crs))
                     for sat_fname, values_l in tqdm(tiff_to_squares.values)]
    full_data =  Parallel(n_jobs=n_jbs)(delayed(subextract_HR_from_tile)(sat_fname=sat_fname,gdf_to_extract=gdf_to_extract) for sat_fname,gdf_to_extract in tqdm(prepare_input))
    #full_data =  [subextract_from_tile(sat_fname,gdf_to_extract) for sat_fname,gdf_to_extract in tqdm(prepare_input)]


def main():
    print("Generating Census")
    geo_df_full_census_data = generate_complete_census_data()
    geo_df_full_census_data = geo_df_full_census_data.to_crs({"init":"epsg:3035"})
    #
    print("Generating UA Boundaries")
    geo_ua_bound = generate_urban_atlas_boundaries()
    geo_census_ua = gpd.sjoin(geo_df_full_census_data,geo_ua_bound)
    #
    print("Generating Satellite Data")
    geo_sat_data = generate_satellite_dataset()
    geo_sat_data = geo_sat_data.to_crs({"init":"epsg:3035"})
    geo_sat_data_cloud_free = geo_sat_data[geo_sat_data["Cloud Cov"]<=2]
    sat_square_to_tiff = gpd.sjoin(geo_census_ua.drop('index_right',axis=1),geo_sat_data_cloud_free[["gis_data","sat_data","geometry"]])
    #
    print("Extracting Labelled Satellite Tiles")
    extract_labelled_imagery(sat_square_to_tiff)

if __name__ == "__main__":
    main()






