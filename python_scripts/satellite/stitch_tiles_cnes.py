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

esa_dir = "/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
dirs_tomerge = [esa_dir + file_dir + "/" + subfile_dir + "/"
               for file_dir in os.listdir(esa_dir)
               if file_dir.endswith('_01')
               for subfile_dir in os.listdir(esa_dir+file_dir)
               if subfile_dir.endswith('_PAN') or subfile_dir.endswith('_MUL')]

for subdir in dirs_tomerge:
    print("Reading %s"%subdir)
    if os.path.isfile(subdir+"merged_output.TIF"):
        print("Skipping file %s"%subdir)
        continue
    try:
        src_files_to_mosaic = [rasterio.open(subdir+x) for x in os.listdir(subdir) if x.endswith(".TIF")]
        print("%d files to merge"%len(src_files_to_mosaic))
        mosaic, out_trans = merge(src_files_to_mosaic)
        print("Finished Merging")
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_crs = src_files_to_mosaic[0].crs["init"]
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
    except:
        print("Error with file: %s"%subdir)
        continue
