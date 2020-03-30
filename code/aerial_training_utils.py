import keras
from keras import backend as K
import pandas as pd
import os
from tqdm import tqdm as tqdmn
import numpy as np
from keras import metrics
import csv
import multiprocessing
from joblib import Parallel, delayed
from skimage import io
import geopandas as gpd
from functools import reduce
import glob
import sys

EPSILON = 1e-10

def precision(y_true, y_pred):
    """Precision metric. Only computes a batch-wise average of precision.
-    Computes the precision, a metric for multi-label classification of
-    how many selected items are relevant.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
-    Only computes a batch-wise average of recall.
-    Computes the recall, a metric for multi-label classification of
-    how many relevant items are selected.
-    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally."""
    return fbeta_score(y_true, y_pred, beta=1)

def my_preprocessor(image):
    # Preprocess images
    image = np.array(image)
    image = (image - np.min(image))/(EPSILON + np.max(image)-np.min(image))
    return image

def chunks(arr, nb_splits):
    #Yield successive n-sized chunks from l.
    order = np.linspace(start=0,stop=len(arr),num=nb_splits + 1)
    for i in range(len(order)-1):
        yield arr[int(order[i]):int(order[i+1])]

def parallel_folder_extraction(im_dir,AERIAL_DIR,null_thresh):
    # Determine which images contain empty pixels
    images = []
    for path in im_dir:
        image = io.imread(AERIAL_DIR + path)
        if  100*(image==0).sum()/image.size > null_thresh :
            images.append((path,False))
        else:
            images.append((path,True))
    return images

def parallel_make_dataset(im_data, CPU_USE, null_thresh = 1):
    # Extract all images in chunks distributed according to CPU_USE
    
    if CPU_USE > 1:
        pre_full = Parallel(n_jobs=CPU_USE)(
            delayed(parallel_folder_extraction)(im_arr,null_thresh=null_thresh)
            for im_arr in tqdmn(chunks(im_data,CPU_USE)))
    else:
        pre_full = [parallel_folder_extraction(im_arr,null_thresh=null_thresh)
                    for im_arr in tqdmn(chunks(im_data,CPU_USE))]
    
    return [data for pre in pre_full for data in pre]
        
def generate_full_idINSPIRE(UA_DIR, AERIAL_DIR, CENSUS_DIR, IMG_OUTPUT_DIR):
    """ Generate all datasets needed for training and gradcaming the tiles"""
    
    # Associate geometry to each census cell
    idINSPIRE2GEOM = gpd.GeoDataFrame.from_file(
        CENSUS_DIR + 'Filosofi2015_carreaux_200m_metropole.shp')[["IdINSPIRE","geometry"]]
    idINSPIRE2GEOM.rename({"IdINSPIRE":"idINSPIRE"},axis=1,inplace=True)
    
    # Associate SES label to each census cell 
    idINSPIRE2SES = pd.read_csv(AERIAL_DIR + "../census_data/squares_to_ses_2019.csv")
    idINSPIRE2SES.rename({"IdINSPIRE":"idINSPIRE"},axis=1,inplace=True)
    idINSPIRE2SES.dropna(subset=["income"],inplace=True)

    # Associate image to each census cell 
    idINSPIRE2VOID = pd.read_csv(AERIAL_DIR + "void_data.csv")
    idINSPIRE2IMG = pd.DataFrame(
        [(im_file.split(".")[0].split("_")[-1],os.path.join(inter_sat_dir,im_file))
         for inter_sat_dir in os.listdir(IMG_OUTPUT_DIR)
         if not inter_sat_dir.endswith(".csv")
         for im_file in os.listdir(IMG_OUTPUT_DIR + inter_sat_dir) 
         if im_file.endswith(".png")], columns = ["idINSPIRE","path2im"])    
    
    # Full Data
    idINSPIRE_full = gpd.GeoDataFrame(
        reduce(lambda left,right: pd.merge(left,right,on=['idINSPIRE']),
               [idINSPIRE2GEOM,idINSPIRE2SES,idINSPIRE2VOID,idINSPIRE2IMG]))
    idINSPIRE_full.crs = idINSPIRE2GEOM.crs
    
    # Filter empty cells
    idINSPIRE_full = idINSPIRE_full[idINSPIRE_full.non_void]

    return idINSPIRE_full