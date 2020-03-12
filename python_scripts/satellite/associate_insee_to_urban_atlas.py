import glob
import pickle
import networkx as nx
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
import warnings
import os, re
from tqdm import tqdm as tqdmn
warnings.filterwarnings("ignore")
from shapely.geometry import MultiLineString
from shapely.ops import polygonize
import json
import seaborn as sns
import numpy as np
import pandas as pd
from shapely import prepared
from sklearn.preprocessing import normalize
from geopandas import GeoDataFrame
import rtree
import sys

# Global paths
BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/esa_URBAN_ATLAS_FR/"
MODEL_OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/model_data/resnet50_keras/"
sys.path.append(BASE_DIR + "SATELSES/equirect_proj_test/cnes/python_files/")
from generate_fr_ua_vhr_data import *


def sjoin(left_df, right_df, how='inner', op='intersects', lsuffix='left', rsuffix='right'):
    index_left = 'index_%s' % lsuffix
    index_right = 'index_%s' % rsuffix
    if (any(left_df.columns.isin([index_left, index_right]))
            or any(right_df.columns.isin([index_left, index_right]))):
        raise ValueError("'{0}' and '{1}' cannot be names in the frames being"
                         " joined".format(index_left, index_right))
    #
    left_df = left_df.copy(deep=True)
    left_df.index = left_df.index.rename(index_left)
    left_df = left_df.reset_index()
    right_df = right_df.copy(deep=True)
    right_df.index = right_df.index.rename(index_right)
    right_df = right_df.reset_index()
    # insert the bounds in the rtree spatial index
    right_df_bounds = right_df.geometry.apply(lambda x: x.bounds)
    stream = ((i, b, None) for i, b in enumerate(right_df_bounds))
    tree_idx = rtree.index.Index(stream)
    idxmatch = (left_df.geometry.apply(lambda x: x.bounds)
                .apply(lambda x: list(tree_idx.intersection(x))))
    #
    one_to_many_idxmatch = idxmatch[idxmatch.apply(len) > 0] 
    if one_to_many_idxmatch.shape[0] > 0:
        r_idx = np.concatenate(one_to_many_idxmatch.values)
        l_idx = np.concatenate([[i] * len(v) for i, v in one_to_many_idxmatch.iteritems()])
        def find_intersects(a1, a2):
            if  a1.intersects(a2):
                return (a1.intersection(a2)).area
            else:
                return 0
        predicate_d = find_intersects
        check_predicates = np.vectorize(find_intersects)         
        result_one_to_many = (pd.DataFrame(np.column_stack([l_idx, r_idx,
                                                            check_predicates(left_df.geometry[l_idx],
                                                            right_df[right_df.geometry.name][r_idx])])))
        result_one_to_many.columns = ['_key_left', '_key_right', 'match_bool']
        result_one_to_many._key_left = result_one_to_many._key_left.astype(int)
        result_one_to_many._key_right = result_one_to_many._key_right.astype(int)
        result_one_to_many = pd.DataFrame(result_one_to_many[result_one_to_many['match_bool'] > 0])
        result_one_to_many = result_one_to_many.groupby("_key_left").apply(
            lambda x : x.ix[np.argmax(x["match_bool"])])
    result = result_one_to_many.set_index('_key_left')
    joined = (
        left_df
        .merge(result, left_index=True, right_index=True)
        .merge(right_df.drop(right_df.geometry.name, axis=1),
               left_on='_key_right', right_index=True,
               suffixes=('_%s' % lsuffix, '_%s' % rsuffix))
    )
    joined = joined.set_index(index_left).drop(['_key_right'], axis=1)
    joined.index.name = None
    return joined

df_car = generate_car_census_data()
poly_dir = glob.glob(UA_DIR + "**/Shapefiles/*UA2012.shp")
full_poly = []
for d in tqdmn(poly_dir):
    test = gpd.read_file(d)
    test["base"] = "/".join(d.split("/")[:-1])+"/"
    full_poly.append(test)

full_poly_shp = gpd.GeoDataFrame(pd.concat(full_poly))
full_poly_shp.crs = full_poly[0].crs
full_poly_shp_car_crs = full_poly_shp.to_crs(df_car.crs)

census2poly = sjoin(df_car,full_poly_shp_car_crs)
census2poly = census2poly[["IDENT","ITEM2012","idINSPIRE","match_bool"]]
print(census2poly.head())
census2poly.to_csv(path_or_buf=UA_DIR + "../insee_to_urban_classdata.csv",sep=";",index=False)

