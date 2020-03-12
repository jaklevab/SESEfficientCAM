import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import os

from keras.datasets import cifar10

from sklearn.preprocessing import MinMaxScaler
import keras
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
from keras import backend as K
from keras.optimizers import SGD, Adam
import pandas as pd
import os
from tqdm import tqdm as tqdmn
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras import metrics
from keras import backend as K
from keras.models import Model
import sys
from skimage import io
import multiprocessing
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def precision(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) 
    precision = true_positives / (predicted_positives + K.epsilon()) 
    return precision 

def recall(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 21))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall 

def fbeta_score(y_true, y_pred, beta=1): 
    if beta < 0: 
        raise ValueError('The lowest choosable beta is zero (only precision).') 
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0: 
        return 0 
    p = precision(y_true, y_pred) 
    r = recall(y_true, y_pred) 
    bb = beta ** 2 
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon()) 
    return fbeta_score 

def fmeasure(y_true, y_pred): 
    return fbeta_score(y_true, y_pred, beta=1) 

img_width, img_height = 32, 32
batch_trainsize=20 #decrease if you machine has low gpu or RAM
batch_testsize=20 #otherwise your code will crash.
nb_epoch = 1

#SGD: Gradient Descent with Momentum and Adaptive Learning Rate
#for more, see here: https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
learningrate=1e-3 #be careful about this parameter. 1e-3 to 1e-8 will train better while learningrate decreases.
momentum=0.8
num_classes = 10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from keras.applications.resnet50 import ResNet50

image_input = Input(shape=(img_width, img_height, 3))
base = ResNet50(input_tensor=image_input, include_top=False,weights='imagenet')
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.inputs, outputs=predictions)
model.compile(optimizer=SGD(lr=learningrate, momentum=momentum),loss="categorical_crossentropy",
              metrics=[fmeasure,recall,precision,"accuracy"])
hist = model.fit(X_train, y_train, batch_size=batch_trainsize, epochs=10, verbose=1
                 ,validation_data=(X_test, y_test))
