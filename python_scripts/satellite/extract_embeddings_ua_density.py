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

# Global paths
BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/esa_URBAN_ATLAS_FR/"
MODEL_OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/model_data/resnet50_keras/UA_density/"

# Global variables
NB_SES_CLASSES = 5
NB_UA_CLASS = 9
PATIENCE_BEFORE_STOPPING = 10
PATIENCE_BEFORE_LOWERING_LR = 2
TRAIN_TEST_FRAC = .8
VAL_SPLIT = .25
BATCH_SIZE = 16
IMG_SIZE = (400, 400)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
MAX_EPOCH = 25
INITIAL_LR = 1e-4
CPU_COUNT = multiprocessing.cpu_count()
CPU_FRAC = .7
CPU_USE = int(CPU_FRAC*CPU_COUNT)
ADRIAN_ALBERT_THRESHOLD = .25
INSEE_AREA = 200*200

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
    """Computes the F score.  
-    The F score is the weighted harmonic mean of precision and recall. 
-    Here it is only computed as a batch-wise average, not globally. 
-    This is useful for multi-label classification, where input samples can be 
-    classified as sets of labels. By only using accuracy (precision) a model 
-    would achieve a perfect score by simply assigning every class to every 
-    input. In order to avoid this, a metric should penalize incorrect class 
-    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0) 
-    computes this, as a weighted mean of the proportion of correct class 
-    assignments vs. the proportion of incorrect class assignments.  
-    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning 
-    correct classes becomes more important, and with beta > 1 the metric is 
-    instead weighted towards penalizing incorrect class assignments. 
-    """ 
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

def parallel_folder_extraction(im_dir,null_thresh):
    images = []
    for path in im_dir:
        image = io.imread(OUTPUT_DIR + path)
        if  100*(image==0).sum()/image.size > null_thresh :
            images.append((path,False))
        else:
            images.append((path,True))
    return images

def parallel_make_dataset(im_data, null_thresh = 1):
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
	    "continuous urban fabric (s.l. > 80%)":"high density urban fabric",
	     "discontinuous dense urban fabric (s.l. 50% - 80%)":"high density urban fabric",
	     "discontinuous medium density urban fabric (s.l. 30% - 50%)":"medium density urban fabric",
	     "discontinuous low density urban fabric (s.l. 10% - 30%)":"low density urban fabric",
	     "discontinuous very low density urban fabric (s.l. < 10%)":"low density urban fabric"
	}
	gdf[class_col] = gdf[class_col].apply(
	    lambda x: consolidate_classes[x] if x in consolidate_classes else x)

	include_classes = ["green urban areas", 
	                   #"airports",
	                   "forests",
	                   "agricultural + semi-natural areas + wetlands",
	                   "high density urban fabric", 
	                   "medium density urban fabric", 
	                   "low density urban fabric",
	                   "water bodies",
                       "water",
	                   "sports and leisure facilities",
	                   "industrial, commercial, public, military and private units"]
	gdf = gdf[gdf[class_col].isin(include_classes)]
	return gdf


print("Loading INSEE SES Dataset")
pre_dic = pd.read_csv(OUTPUT_DIR + "../census_data/squares_to_ses.csv" )
pre_dic.dropna(subset=["income"],inplace=True)

print("Loading image")
image_files = [os.path.join(inter_sat_dir,im_file) for inter_sat_dir in tqdmn(os.listdir(OUTPUT_DIR) )
               for im_file in os.listdir(OUTPUT_DIR + inter_sat_dir) if im_file.endswith(".png")]
im_df = pd.DataFrame()
im_df["path2im"] = image_files
im_df["idINSPIRE"] = [k.split("/")[-1].split(".")[0].split("_")[-1] for k in image_files]
full_im_df = pd.merge(pre_dic,im_df,on="idINSPIRE")

print("Filtering Void Datasets")
check_void = parallel_make_dataset(full_im_df.path2im)
void_df = pd.DataFrame(check_void,columns=["path2im","non_void"])
full_im_df = pd.merge(full_im_df,void_df,on="path2im")
full_im_df = full_im_df[full_im_df.non_void]
full_im_df.reset_index(drop=True,inplace=True)

print("Loading UA Classes")
insee2ua = pd.read_csv(UA_DIR + "../insee_to_max_urban_class_data.csv",sep=";")
treat_class = lambda x :x.lower()\
     .replace(":","")\
     .replace("  "," ")\
     .replace("…","...")\
     .replace(", ("," (")\
     .replace("vegetations","vegetation")

insee2ua["right_class"] = [treat_class(x) for x in insee2ua["right_class"]]
insee2ua = consolidate_UA_classes(insee2ua,"right_class")
insee2ua = insee2ua[insee2ua.match_bool > (ADRIAN_ALBERT_THRESHOLD*INSEE_AREA)] #contain at least 25% of associated ground truth polygon
full_im_df_ua = pd.merge(full_im_df,insee2ua[["right_class","idINSPIRE"]],on="idINSPIRE")

print("Generating Density Classes")
density = full_im_df_ua.ind_c
class_thresholds = [np.percentile(density,k) for k in np.linspace(0,100,NB_SES_CLASSES +1 )]
x_to_class = np.digitize(density,class_thresholds)
x_to_class[x_to_class==np.max(x_to_class)] = NB_SES_CLASSES
full_im_df_ua["treated_density"] = [ str(y-1) for y in x_to_class ]

print("Generating Density+UA Classes")
full_im_df_ua["proxy_class"] = [(x,y) for x,y in full_im_df_ua[["treated_density","right_class"]].values]

print("Generating Generators")
full_im_df_ua = full_im_df_ua.sample(frac=1)
full_datagen = ImageDataGenerator(preprocessing_function=my_preprocessor)

k_samples = 100000
fully_sampled = full_im_df_ua.head(k_samples)
full_generator = full_datagen.flow_from_dataframe(
        fully_sampled,
        directory=OUTPUT_DIR,
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

fully_sampled["cx"] = tsne_embeddings[:,0]
fully_sampled["cy"] = tsne_embeddings[:,1]
fully_sampled.to_csv(MODEL_OUTPUT_DIR + "embeddings/tsne_emb.csv",index=False)

# Clear model from GPU after each iteration
K.clear_session()
