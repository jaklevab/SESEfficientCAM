# General Imports
import sys
import os
from time import time
import rasterio
import pyproj
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Point
import warnings
import argparse
import pdb
from rasterio.merge import merge

ign_dir = "/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/ign/ORTHO_HR"

dirs_to_merge = [ign_dir + "/" + file_dir + "/" + first_subdir + "/" + second_subdir + "/" + third_dir + "/" + fourth_dir + "/"
               for file_dir in os.listdir(ign_dir) #departments
               for first_subdir in os.listdir(ign_dir + "/" + file_dir) #single folder
               for second_subdir in os.listdir(ign_dir + "/" + file_dir + "/" + first_subdir) #single folder (ORTHOHR)
               for third_dir in os.listdir(ign_dir + "/" + file_dir + "/" + first_subdir + "/" + second_subdir) #one of three subdirs
               if third_dir.startswith('1_DONNEES_LIVRAISON_')
               for fourth_dir in os.listdir(ign_dir + "/" + file_dir + "/" + first_subdir + "/" + second_subdir + "/" + third_dir) #one
               if not fourth_dir.endswith(".md5")
               ]

for subdir in dirs_to_merge:
    files_to_merge = [subdir + file for file in os.listdir(subdir) if file.endswith('.jp2') ]    
    try:
        src_files_to_mosaic = [rasterio.open(x) for x in files_to_merge ]
        print("%d files to merge"%len(src_files_to_mosaic))
        mosaic, out_trans = merge(src_files_to_mosaic)
        print("Finished Merging")
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_crs = src_files_to_mosaic[0].crs#["init"]
        out_meta.update({"driver": "GTiff",
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans,
                         "crs": out_crs,
                        })
        print("Writing merged to output")
        with rasterio.open(subdir+"merged_output.TIF", "w", **out_meta) as dest:
            dest.write(mosaic)
        print("Close all files")
        for src in src_files_to_mosaic:
            src.close()
    except Exception as e:
        print("Error with file: %s"%subdir)
        print(str(e))
        continue
