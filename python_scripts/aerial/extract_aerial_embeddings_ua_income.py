from MulticoreTSNE import MulticoreTSNE as TSNE
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
from keras.preprocessing.balanced_image import BalancedImageDataGenerator
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
from efficientnet import EfficientNetB0 as EfficientNet
import geopandas as gpd 
import sys
sys.path.append("/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/")
from aerial_training_utils import generate_full_idINSPIRE, geographical_boundaries, my_preprocessor, fmeasure,recall,precision, fbeta_score

# Global paths
BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
MODEL_OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/model_data/efficientnet_keras/UA_income/"
BASE_AERDIR = BASE_DIR + "SATELSES/"
AERIAL_DIR= BASE_AERDIR + "equirect_proj_test/cnes/data_files/outputs/AERIAL_esa_URBAN_ATLAS_FR/"

# Global variables
NB_SES_CLASSES = 5
NB_UA_CLASS = 11
PATIENCE_BEFORE_STOPPING = 10
PATIENCE_BEFORE_LOWERING_LR = 2
TRAIN_TEST_FRAC = .8
VAL_SPLIT = .25
BATCH_SIZE = 10 #16
IMG_SIZE = (800, 800)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
MAX_EPOCH = 25
INITIAL_LR = 1e-4
CPU_COUNT = multiprocessing.cpu_count()
CPU_FRAC = .7
CPU_USE = int(CPU_FRAC*CPU_COUNT)
ADRIAN_ALBERT_THRESHOLD = .25
INSEE_AREA = 200*200


print("Generating Income+UA Classes")
full_im_df_ua = generate_full_idINSPIRE(UA_DIR, AERIAL_DIR, NB_SES_CLASSES, ADRIAN_ALBERT_THRESHOLD, INSEE_AREA)
full_im_df_ua["proxy_class"] = [(x,y) for x,y in full_im_df_ua[["treated_income","right_class"]].values]

print("Generating Generators")
print("Generating Generators")
full_im_df_ua = full_im_df_ua.sample(frac=1)
full_datagen = ImageDataGenerator(preprocessing_function=my_preprocessor)

k_samples = 75000
fully_sampled = full_im_df_ua.head(k_samples)
full_generator = full_datagen.flow_from_dataframe(
        fully_sampled,
        directory=AERIAL_DIR,
        x_col="path2im",
        y_col="proxy_class",
        target_size=IMG_SIZE,
        color_mode ="rgb",
        shuffle=False,
        batch_size=1,
        interpolation="bicubic",
        subset="training",
        class_mode='categorical')

# Load the last best model
dic_load_model = {
    "precision":precision,
    "recall":recall,
    "fbeta_score":fbeta_score,
    "fmeasure":fmeasure
}
model = load_model(
    MODEL_OUTPUT_DIR + "lastbest-0.hdf5",
    custom_objects=dic_load_model)

emb = Model(inputs=model.input,outputs=model.get_layer("global_average_pooling2d_1").output)
print("Predicting Generator")
data_embeddings = emb.predict_generator(full_generator,steps=k_samples,verbose=1)
print("TSNE")
tsne_embeddings = TSNE(n_jobs=20).fit_transform(data_embeddings)

fully_sampled["ori"] = data_embeddings.tolist()
fully_sampled["cx"] = tsne_embeddings[:,0]
fully_sampled["cy"] = tsne_embeddings[:,1]
fully_sampled.to_csv(MODEL_OUTPUT_DIR + "embeddings/aerial_full_tsne_emb.csv",index=False)

# Clear model from GPU after each iteration
K.clear_session()
