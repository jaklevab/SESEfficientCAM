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

BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/esa_URBAN_ATLAS_FR/"
MAX_NB_JOBS = 20

def generate_rect_census_data():
    '''Generate Rectangular Census Data'''
    df_rec = GeoDataFrame.from_file(CENSUS_DIR + "200m-rectangles-metropole/rect_m.dbf")
    df_rec_data = GeoDataFrame.from_file(CENSUS_DIR + "200m-rectangles-metropole/rect_m.mif")
    df_rec_final=pd.merge(df_rec,df_rec_data,how="inner",on="idk")
    df_rec_final.drop("geometry_x",inplace=True,axis=1)
    df_rec_final["income"] = df_rec_final.ind_srf/df_rec_final.ind_r
    df_rec_final["owner_ratio"] = df_rec_final.men_prop/df_rec_final.men
    df_rec_final["pov_rate"] = df_rec_final.men_basr/df_rec_final.men
    df_rec_final.crs = {'proj': 'lcc',
         'lat_1': 45.90287723937,
         'lat_2': 47.69712276063,
         'lat_0': 46.8,
         'lon_0': 2.337229104484,
         'x_0': 600000,
         'y_0': 2200000,
         'ellps': 'clrk80',
         'towgs84': '-168,-60,320,0,0,0,0',
         'units': 'm',
         'no_defs': True}
    return df_rec_final

def generate_car_census_data():
    '''Generate Square Census Data'''
    df_car = GeoDataFrame.from_file(CENSUS_DIR + "200m-carreaux-metropole/car_m.dbf")
    df_car_data = GeoDataFrame.from_file(CENSUS_DIR + "200m-carreaux-metropole/car_m.mif")
    df_car_final=pd.merge(df_car_data,df_car,how="inner",on="idINSPIRE")
    df_car_final.drop(["id_y","geometry_y"],inplace=True,axis=1)
    df_car_final.columns = ['idINSPIRE', 'id','geometry','idk','ind_c','nbcar']
    df_car_final["density"]=df_car_final.ind_c/0.04
    return df_car_final

def generate_complete_census_data():
    df_rec = generate_rect_census_data()
    df_car = generate_car_census_data()
    df_final = pd.merge(df_car,df_rec,how="inner",on="idk",suffixes=('_car','_rec'))
    df_final.rename({"geometry_car":"geometry"},axis=1,inplace=True)
    geo_df_final = gpd.GeoDataFrame(df_final)
    geo_df_final.crs = df_car.crs
    return geo_df_final

def generate_new_census_data():
    df_rec = GeoDataFrame.from_file(BASE_DIR + 'INSEE/2019/200m/shps/Filosofi2015_carreaux_200m_metropole.shp')
    df_rec["income"] = df_rec["Ind_snv"]/df_rec["Ind"]
    df_rec["pov_rate"] = df_rec["Men_pauv"]/df_rec["Men"]
    return df_rec

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def generate_urban_atlas_boundaries():
    ''' Generates Boundaries for each city in Urban Atlas'''
    ua_bounds = [(ua.split("_")[-1],gpd.read_file(UA_DIR + ua +"/Shapefiles/" + x).geometry.values[0])
                  for ua in os.listdir(UA_DIR) for x in os.listdir(UA_DIR + ua + "/Shapefiles/")
                 if x.endswith("_UA2012_Boundary.shp")]
    ua_gdf = gpd.GeoDataFrame(ua_bounds,columns=["city","geometry"])
    ua_gdf.crs = {"init":"epsg:3035"}
    return ua_gdf

def generate_satellite_dataset():
    """ Generate goto df with geo and metadata information for availabel satellite imagery"""
    sat_info = pd.read_csv("/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/sources/FR_UA.csv",sep=";")
    sat_info["data_dir"] = [k.split(":")[-1]+"_01" for k in sat_info["Eop Id"].values]
    gis_data = [[SAT_DIR + in_dir + "/GIS_FILES/" + gis for gis in os.listdir(SAT_DIR + in_dir + "/GIS_FILES")
                 if gis.endswith("_PIXEL_SHAPE.shp")][0]
                for in_dir in os.listdir(SAT_DIR) if in_dir.endswith("_01") and
                "GIS_FILES" in os.listdir(SAT_DIR + in_dir )]
    sat_data = ["/".join(d.split("/")[:-2]) + "/merged_pansharpened_corrected_projected.TIF" 
                if "merged_pansharpened_corrected_projected.TIF" in os.listdir("/".join(d.split("/")[:-2]) + "/")
                else "0" for d in gis_data]
    df_to_extract = pd.DataFrame()
    df_to_extract["gis_data"] = gis_data
    df_to_extract["sat_data"] = sat_data
    df_to_extract = df_to_extract[df_to_extract.sat_data!="0"]
    df_to_extract["data_dir"] = [k.split("/")[-2] for k in df_to_extract["sat_data"].values]
    df_to_extract["geometry"] = [gpd.read_file(x).geometry.values[0] for x in df_to_extract.gis_data]
    df_to_extract = pd.merge(df_to_extract,sat_info[["data_dir","Cloud Cov"]],on="data_dir")
    geo_df_to_extract = gpd.GeoDataFrame(df_to_extract)
    geo_df_to_extract.crs = {"init":"epsg:4326"}
    return  geo_df_to_extract

def subextract_from_tile(sat_fname,gdf_to_extract):
    sat_data = rasterio.open(sat_fname)
    curr_sat_dir = sat_fname.split("/")[-2]
    if not os.path.exists(OUTPUT_DIR + curr_sat_dir):
        os.makedirs(OUTPUT_DIR + curr_sat_dir)
    #
    for it in range(gdf_to_extract.shape[0]):
        geo = gdf_to_extract.iloc[it:(it+1)].to_crs(sat_data.crs)
        coords = getFeatures(geo)
        _, idINSPIRE = geo.values[0]
        out_img, out_transform = rasterio.mask.mask(dataset=sat_data, shapes=coords, crop=True)
        reorder_img = np.dstack([out_img[0],out_img[1],out_img[2]])
        io.imsave(OUTPUT_DIR + curr_sat_dir + "/FR_URBANATLAS_200m_%s.png"%idINSPIRE,skimage.img_as_ubyte(reorder_img))
    sat_data.close()
    print("Done")
    return None

def extract_labelled_imagery(sat_square_to_tiff):
    tiff_to_squares = sat_square_to_tiff.groupby("sat_data")[["geometry","idINSPIRE"]].apply(lambda x : list(x.values)).reset_index()
    nb_tiles = tiff_to_squares.shape[0]
    n_jbs = min(nb_tiles, MAX_NB_JOBS)
    prepare_input = [(sat_fname,gpd.GeoDataFrame(pd.DataFrame(values_l,columns=["geometry","idINSPIRE"]),crs=sat_square_to_tiff.crs))
                     for sat_fname, values_l in tqdm(tiff_to_squares.values)]
    full_data =  Parallel(n_jobs=n_jbs)(delayed(subextract_from_tile)(sat_fname=sat_fname,gdf_to_extract=gdf_to_extract) for sat_fname,gdf_to_extract in tqdm(prepare_input))
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






