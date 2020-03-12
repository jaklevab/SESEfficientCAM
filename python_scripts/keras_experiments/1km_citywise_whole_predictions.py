import keras
from keras import backend as K
from keras.optimizers import SGD, Adam
import pandas as pd
import cv2
import os,sys
from tqdm import tqdm as tqdmn
import numpy as np
from keras import metrics
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Input
from skimage import io
import glob
import tensorflow as tf
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
from functools import reduce
from scipy.stats import entropy
from efficientnet import EfficientNetB0 as EfficientNet
import pickle
from keras import backend as K
import glob 
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Concatenate, Input, Lambda, Multiply
from scipy.special import binom
from sklearn.model_selection import StratifiedKFold

BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/"
OUTPUT_DIR = BASE_DIR + "cnes/data_files/outputs/1km_AERIAL_esa_URBAN_ATLAS_FR/"
MODEL_OUTPUT_DIR = BASE_DIR + "cnes/data_files/outputs/model_data/efficientnet_keras/"
RES_DIR = MODEL_OUTPUT_DIR + "1km_2019_income_norm_v2/"

def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision 

def recall(y_true, y_pred): 
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

IMG_SIZE = (800, 800)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
NB_SES_CLASSES = 5

dic_load_model = {
    "precision":precision,"recall":recall,"fbeta_score":fbeta_score,"fmeasure":fmeasure,
    "binom":binom,"Multiply":Multiply,"Concatenate":Concatenate,"Lambda":Lambda,"NB_SES_CLASSES":NB_SES_CLASSES,
}

city_assoc = pd.read_csv(OUTPUT_DIR + "city_assoc.csv")
idINSPIRE2VOID = pd.read_csv(OUTPUT_DIR + "void_data.csv")
idINSPIRE2IMG = pd.DataFrame(
    [(im_file.split(".")[0].split("_")[-1],os.path.join(inter_sat_dir,im_file))
     for inter_sat_dir in os.listdir(OUTPUT_DIR) if not inter_sat_dir.endswith(".csv")
     for im_file in os.listdir(OUTPUT_DIR + inter_sat_dir) if im_file.endswith(".png")],
    columns = ["Id_carr1km","path2im"])
idINSPIRE_full = pd.DataFrame(reduce(lambda left,right: pd.merge(left,right,on=['Id_carr1km']),
                                      [idINSPIRE2VOID,idINSPIRE2IMG,city_assoc]))
idINSPIRE_full = idINSPIRE_full[idINSPIRE_full.non_void]
cities = {
    city:pd.concat([pd.read_csv(fold_file,header=0,sep=",")
                    for fold_file in glob.glob(RES_DIR+city+"/*last_best_models.csv")], axis=0).reset_index(drop=True)
    for city in os.listdir(RES_DIR) 
    if "1km_{}_income_pred.pdf".format(city) not in os.listdir("/warehouse/COMPLEXNET/jlevyabi/tmp/")
    and len(glob.glob(RES_DIR+city+"/*last_best_models.csv"))>0
}
best_model_city = {k:v.ix[v["Validation loss"].idxmin()]["Model file"] for k,v in cities.items()}
print(best_model_city)
for city, best_model in best_model_city.items():
    if "1km_full_whole_predicted_values.csv" in os.listdir("{}{}/preds/".format(RES_DIR,city)):
        print("Skipping {}".format(city))
        continue
    print("Predicting {} with model weight file {}".format(city,best_model))
    current_city_assoc = idINSPIRE_full[idINSPIRE_full.FUA_NAME == city]
    current_city_assoc["treated_citywise_income"] = np.random.choice(a=[str(k) for k in range(NB_SES_CLASSES)],
                                                                     p=[.2,.2,.2,.2,.2],
                                                                     size=current_city_assoc.shape[0]).tolist()
    efficientnet_model = load_model("{}{}/{}".format(RES_DIR,city,best_model_city[city]),custom_objects=dic_load_model)
    city_datagen = ImageDataGenerator(preprocessing_function=my_preprocessor)
    city_generator = city_datagen.flow_from_dataframe(
            dataframe=current_city_assoc,
            directory=OUTPUT_DIR,
            x_col="path2im",
            y_col="treated_citywise_income",
            target_size=IMG_SIZE,
            color_mode ="rgb",
            shuffle=False,
            batch_size=1,
            interpolation="bicubic",
            class_mode='categorical')
    ses_predictions = efficientnet_model.predict_generator(city_generator,current_city_assoc.shape[0], workers=20, verbose=1)
    y_pred_ses = np.argmax(ses_predictions, axis=1)
    for k in range(NB_SES_CLASSES):
        current_city_assoc["pred_val_{}".format(k)] = ses_predictions[:,k]
    current_city_assoc["out"] = y_pred_ses
    current_city_assoc[["Id_carr1km","out",]+["pred_val_{}".format(k) for k in range(NB_SES_CLASSES)]].to_csv("{}{}/preds/1km_full_whole_predicted_values.csv".format(RES_DIR,city),index=False)

