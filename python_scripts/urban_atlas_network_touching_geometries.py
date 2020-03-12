import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input")
args = parser.parse_args()
input_file = args.input
city = input_file.split("_")[-1].lower()
base_dir = "/warehouse/COMPLEXNET/jlevyabi/"
ua_dir = base_dir + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/" + input_file + "/Shapefiles/"
city_ua = gpd.read_file(ua_dir + input_file + "_UA2012.shp")

def generate_neigh_dic(set_of_nodes):
    dic_net = {}
    for index, poly in city_ua[["IDENT","geometry"]].iloc[set_of_nodes].iterrows():
        # get 'not disjoint' countries
        neighbors = city_ua[~city_ua.geometry.disjoint(poly.geometry)].IDENT.tolist()
        # remove own name from the list
        dic_net[poly.IDENT] = [ name for name in neighbors if poly.IDENT != name]
    return dic_net

def generate_full_dic(arr_dic):
    init_dic = arr_dic[0]
    for d in tqdm(arr_dic[1:]):
         init_dic.update(d)
    return init_dic

def main():
    nb_pols = city_ua.shape[0]
    #
    ideal_workload = step = 1000
    max_nb_jobs = 60
    n_jbs = min(nb_pols//ideal_workload, max_nb_jobs)
    print("%d polygons to sift through"%nb_pols)
    print("Partitioning and Computing among %d cores"%n_jbs)
    pre_full_info = Parallel(n_jobs = n_jbs)(
        delayed(generate_neigh_dic)(list(range(idx,min(idx + step,nb_pols))))
        for idx in tqdm(range(0, nb_pols, step)))
    #
    print("Stitching")
    full_info = generate_full_dic(pre_full_info)
    #
    print("Outputting")
    pickle.dump(full_info,open(ua_dir + "../" + city + "_ua_network.p","wb"))

if __name__ == "__main__":
    main()

