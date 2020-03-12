from sklearn.preprocessing import MinMaxScaler
import keras
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Concatenate
from keras import backend as K
from keras.optimizers import SGD, Adam
import pandas as pd
import os
from tqdm import tqdm as tqdmn
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.balanced_image import BalancedImageDataGenerator
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras import metrics
from keras import backend as K
from keras.models import Model, load_model
import csv
from sklearn.metrics import confusion_matrix, classification_report
from time import time
from keras.layers import Input
import multiprocessing
from joblib import Parallel, delayed
from skimage import io
#from efficientnet import EfficientNetB5 as EfficientNet
import geopandas as gpd
from functools import reduce
import glob
import sys
sys.path.append("/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/satellite/")
from generate_fr_ua_vhr_data import generate_car_census_data, generate_new_census_data

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def top_k_accuracy(y_true, y_pred,val_k):
    """ UNTESTED """
    topk_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=val_k)
    topk_acc.__name__ = 'topk_acc'
    return topk_acc(y_true, y_pred)

def neigh_k_accuracy(y_true, y_pred, val_k):
    """ UNTESTED """
    kernel = K.ones((1,val_k),)
    classes_pred = K.conv1d(y_pred,kernel,padding='same')
    true_positives = K.sum(K.round(K.clip(y_true * classes_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    return(true_positives / (predicted_positives + K.epsilon()))  

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
    Here it is only computed as a batch-wise average, not globally. 
    """ 
    return fbeta_score(y_true, y_pred, beta=1) 

def my_preprocessor(image):
    image = np.array(image)
    image = (image - np.min(image))/(.1 + np.max(image)-np.min(image))
    return image

def chunks(arr, nb_splits):
    #Yield successive n-sized chunks from l.
    order = np.linspace(start=0,stop=len(arr),num=nb_splits + 1)
    for i in range(len(order)-1):
        yield arr[int(order[i]):int(order[i+1])]

def parallel_folder_extraction(im_dir,AERIAL_DIR,null_thresh):
    images = []
    for path in im_dir:
        image = io.imread(AERIAL_DIR + path)
        if  100*(image==0).sum()/image.size > null_thresh :
            images.append((path,False))
        else:
            images.append((path,True))
    return images

def parallel_make_dataset(im_data, CPU_USE, null_thresh = 1):
    nb_images = len(im_data)
    pre_full = Parallel(n_jobs=CPU_USE)(
        delayed(parallel_folder_extraction)(im_arr,null_thresh=null_thresh)
        for im_arr in tqdmn(chunks(im_data,CPU_USE)))
    return [data for pre in pre_full for data in pre]

def consolidate_UA_classes(gdf, class_col='ITEM'):
	consolidate_classes = {
        "arable land (annual crops)":"agricultural + semi-natural areas + wetlands",
        "permanent crops (vineyards, fruit trees, olive groves)":"agricultural + semi-natural areas + wetlands",
        "pastures":"agricultural + semi-natural areas + wetlands",
        "water bodies":"water",
        "complex and mixed cultivation patterns":"agricultural + semi-natural areas + wetlands",
        "orchards":"agricultural + semi-natural areas + wetlands",
	    "continuous urban fabric (s.l. > 80%)":"very high density urban fabric",
	     "discontinuous dense urban fabric (s.l. 50% - 80%)":"high density urban fabric",
	     "discontinuous medium density urban fabric (s.l. 30% - 50%)":"medium density urban fabric",
	     "discontinuous low density urban fabric (s.l. 10% - 30%)":"low density urban fabric",
	     "discontinuous very low density urban fabric (s.l. < 10%)":"very low density urban fabric"
	}
	gdf[class_col] = gdf[class_col].apply(
	    lambda x: consolidate_classes[x] if x in consolidate_classes else x)

	include_classes = ["green urban areas", 
	                   "forests",
	                   "agricultural + semi-natural areas + wetlands",
	                   "very high density urban fabric", 
                       "high density urban fabric", 
	                   "medium density urban fabric", 
	                   "low density urban fabric",
                       "very low density urban fabric",
                       "water",
	                   "sports and leisure facilities",
	                   "industrial, commercial, public, military and private units"]
	gdf = gdf[gdf[class_col].isin(include_classes)]
	return gdf

def generate_full_idINSPIRE(UA_DIR, AERIAL_DIR, NB_SES_CLASSES, ADRIAN_ALBERT_THRESHOLD=.25, INSEE_AREA=200*200, old=True):
    # Geom Data
    if old:
        idINSPIRE2GEOM = generate_car_census_data()[["idINSPIRE","geometry"]]
    else:
        idINSPIRE2GEOM = generate_new_census_data()[["IdINSPIRE","geometry"]]
        idINSPIRE2GEOM.rename({"IdINSPIRE":"idINSPIRE"},axis=1,inplace=True)
    # UA Data
    idINSPIRE2UA = pd.read_csv(UA_DIR + "../insee_to_max_urban_class_data.csv",sep=";")
    treat_class = lambda x :x.lower()\
         .replace(":","")\
         .replace("  "," ")\
         .replace("…","...")\
         .replace(", ("," (")\
         .replace("vegetations","vegetation")
    idINSPIRE2UA["right_class"] = [treat_class(x) for x in idINSPIRE2UA["right_class"]]
    idINSPIRE2UA = consolidate_UA_classes(idINSPIRE2UA,"right_class")
    idINSPIRE2UA = idINSPIRE2UA[idINSPIRE2UA.match_bool > (ADRIAN_ALBERT_THRESHOLD*INSEE_AREA)] 
    #
    # SES Data
    if old:
        idINSPIRE2SES = pd.read_csv(AERIAL_DIR + "../census_data/squares_to_ses.csv")
    else:
        idINSPIRE2SES = pd.read_csv(AERIAL_DIR + "../census_data/squares_to_ses_2019.csv")
        idINSPIRE2SES.rename({"IdINSPIRE":"idINSPIRE"},axis=1,inplace=True)
    idINSPIRE2SES.dropna(subset=["income"],inplace=True)
    income = idINSPIRE2SES.income
    class_thresholds = [np.percentile(income,k) for k in np.linspace(0,100,NB_SES_CLASSES +1 )]
    x_to_class = np.digitize(income,class_thresholds)
    x_to_class[x_to_class==np.max(x_to_class)] = NB_SES_CLASSES
    idINSPIRE2SES["treated_income"] = [ str(y-1) for y in x_to_class ]
    #
    # IMG Data
    idINSPIRE2VOID = pd.read_csv(AERIAL_DIR + "void_data.csv")
    idINSPIRE2IMG = pd.DataFrame(
        [(im_file.split(".")[0].split("_")[-1],os.path.join(inter_sat_dir,im_file))
         for inter_sat_dir in os.listdir(AERIAL_DIR) if not inter_sat_dir.endswith(".csv")
         for im_file in os.listdir(AERIAL_DIR + inter_sat_dir) if im_file.endswith(".png")],
        columns = ["idINSPIRE","path2im"])
    #
    # Full Data
    idINSPIRE_full = gpd.GeoDataFrame(
        reduce(lambda left,right: pd.merge(left,right,on=['idINSPIRE']),
               [idINSPIRE2GEOM,idINSPIRE2UA,idINSPIRE2SES,idINSPIRE2VOID,idINSPIRE2IMG]))
    idINSPIRE_full.crs = idINSPIRE2GEOM.crs
    idINSPIRE_full = idINSPIRE_full[idINSPIRE_full.non_void]
    #
    return idINSPIRE_full

def generate_full_idINSPIRE_1km(BASE_DIR,AERIAL_DIR, NB_SES_CLASSES, ):
    idINSPIRE2GEOM = gpd.read_file(BASE_DIR+"INSEE/2019/1km/shps/Filosofi2015_carreaux_1000m_metropole.shp")
    idINSPIRE2SES = pd.read_csv(AERIAL_DIR + "../census_data/1km_squares_to_ses_2019.csv")
    idINSPIRE2SES.dropna(subset=["income"],inplace=True)
    income = idINSPIRE2SES.income
    class_thresholds = [np.percentile(income,k) for k in np.linspace(0,100,NB_SES_CLASSES +1 )]
    x_to_class = np.digitize(income,class_thresholds)
    x_to_class[x_to_class==np.max(x_to_class)] = NB_SES_CLASSES
    idINSPIRE2SES["treated_income"] = [ str(y-1) for y in x_to_class ]
    #
    # IMG Data
    idINSPIRE2VOID = pd.read_csv(AERIAL_DIR + "void_data.csv")
    idINSPIRE2IMG = pd.DataFrame(
        [(im_file.split(".")[0].split("_")[-1],os.path.join(inter_sat_dir,im_file))
         for inter_sat_dir in os.listdir(AERIAL_DIR) if not inter_sat_dir.endswith(".csv")
         for im_file in os.listdir(AERIAL_DIR + inter_sat_dir) if im_file.endswith(".png")],
        columns = ["Id_carr1km","path2im"])
    #
    # Full Data
    idINSPIRE_full = gpd.GeoDataFrame(
        reduce(lambda left,right: pd.merge(left,right,on=['Id_carr1km']),
               [idINSPIRE2GEOM,idINSPIRE2SES,idINSPIRE2VOID,idINSPIRE2IMG]))
    idINSPIRE_full.crs = idINSPIRE2GEOM.crs
    idINSPIRE_full = idINSPIRE_full[idINSPIRE_full.non_void]
    return idINSPIRE_full


def geographical_boundaries(UA_DIR):
    UA_POLY_DIR = glob.glob(UA_DIR + "**/Shapefiles/*UA2012_Boundary.shp")
    CITY_POLY_DIR = glob.glob(UA_DIR + "**/Shapefiles/*CityBoundary.shp")
    UA_COLS = ["AREA_KM2","COUNTRY","FUA_NAME","URBAN_KM2","URBAN_RATE","LU12_AVAIL","LU12_DATE","Pop2012","geometry"]
    CITY_COLS = ["URAU_NAME","POPL_2015","geometry"]
    UABOUNDARY = gpd.GeoDataFrame(pd.concat([gpd.read_file(d)[UA_COLS] for d in UA_POLY_DIR],sort=True))
    UABOUNDARY.crs = {'init': 'epsg:3035'}
    CITYBOUNDARY = gpd.GeoDataFrame(pd.concat([gpd.read_file(d)[CITY_COLS] for d in CITY_POLY_DIR],sort=True))
    CITYBOUNDARY.crs = {'init': 'epsg:3035'}
    return (UABOUNDARY, CITYBOUNDARY)

