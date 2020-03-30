import pandas as pd
import cv2
import geopandas as gpd
import os,sys
from tqdm import tqdm as tqdmn
import numpy as np
import pickle
import argparse
from scipy.special import binom
import multiprocessing
from joblib import Parallel, delayed
from functools import reduce
from scipy.stats import entropy
from skimage import io

import tensorflow as tf
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Input
from keras.preprocessing import image
from tensorflow.python.framework import ops
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Concatenate, Input, Lambda, Multiply
from keras import backend as K
from keras.optimizers import SGD, Adam
from efficientnet.keras import EfficientNetB0 as EfficientNet

from rasterio.transform import from_bounds
from rasterstats import zonal_stats, point_query, gen_zonal_stats
import rasterio
import rtree

import glob
from aerial_training_utils import generate_full_idINSPIRE, my_preprocessor, fmeasure,recall,precision, fbeta_score

# Global paths
DATA_BASE_DIR = "../data/"
OUTPUT_BASE_DIR = "../results/"
AERIAL_DIR = DATA_BASE_DIR + "aerial_data/"
CENSUS_DIR = DATA_BASE_DIR + 'census_data/'
UA_DIR = DATA_BASE_DIR + "UA_data/"
IMG_OUTPUT_DIR = OUTPUT_BASE_DIR + "imagery_out/"
MODEL_OUTPUT_DIR = OUTPUT_BASE_DIR + "model_data/"

# Global variables

# SES/UA Related
NB_SES_CLASSES = 5
EPSILON = 1e-10

# Image Related
W = H = 800
IMG_SIZE = (W, H)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
DIC_LOAD_MODEL = {"precision":precision, "recall":recall, "fbeta_score":fbeta_score, "fmeasure":fmeasure, "binom":binom,
                  "Multiply":Multiply, "Concatenate":Concatenate, "Lambda":Lambda, "NB_SES_CLASSES":NB_SES_CLASSES}

# Model Related
conv_name = 'top_conv'
input_name = 'input_1'

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('-city','--city',default="Paris",help = 'City to study')
parser.add_argument('-max_bs','--max_bs',help = 'Batch Size', type=int, default=15)
parser.add_argument('-workload','--workload', type=int, default=4000)

args = parser.parse_args()
WORKLOAD = args.workload #(Outer Batch Size)
MAX_BS = args.max_bs #(Inner Batch Size )
city = args.city

# GPU/Keras/TF configuration setup
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

#Base class for saliency masks. Alone, this class doesn't do anything.
class SaliencyMask(object):
    """Base class for saliency masks. Alone, this class doesn't do anything."""
    def __init__(self, model, output_index=0):
        """Constructs a SaliencyMask.
        Args:
            model: the keras model used to make prediction
            output_index: the index of the node in the last layer to take derivative on
        """
        pass

    def get_mask(self, input_image):
        """Returns an unsmoothed mask.
        Args:
            input_image: input image with shape (H, W, 3).
        """
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=1, nsamples=100):
        """Returns a mask that is smoothed with the SmoothGrad method.
        Args:
            input_image: input image with shape (H, W, 3).
        """
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))
        total_gradients = np.zeros_like(input_image)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise
            total_gradients += self.get_mask(x_value_plus_noise)
        return total_gradients / nsamples

#A SaliencyMask class that computes saliency masks with a gradient.
class GradientSaliency(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with a gradient."""
    def __init__(self, model, output_index=0):
        # Define the function to compute the gradient
        input_tensors = [model.input,        # placeholder for input image tensor
                         K.learning_phase(), # placeholder for mode (train or test) tense
                        ]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    def get_mask(self, input_image):
        """Returns a vanilla gradient mask.
        Args:
            input_image: input image with shape (H, W, 3).
        """
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value, 0])[0][0]
        return gradients

#A SaliencyMask class that computes saliency masks with GuidedBackProp.
class GuidedBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with GuidedBackProp.
    This implementation copies the TensorFlow graph to a new graph with the ReLU
    gradient overwritten as in the paper: https://arxiv.org/abs/1412.6806
    """
    GuidedReluRegistered = False
    def __init__(self, model, output_index=0, custom_loss=None):
        #
        model_save_loc = '/tmp/gb_keras_gpu_{}_{}.h5'.format(city,output_index)
        session_save_loc = '/tmp/guided_backprop_ckpt_gpu_{}_{}'.format(city,output_index)
        graph_save_loc = '/tmp/guided_backprop_ckpt_gpu_{}_{}.meta'.format(
            city,output_index)
        #
        """Constructs a GuidedBackprop SaliencyMask."""
        if GuidedBackprop.GuidedReluRegistered is False:
            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0, "float32")
                gate_y = tf.cast(op.outputs[0] > 0, "float32")
                return gate_y * gate_g * grad
        GuidedBackprop.GuidedReluRegistered = True
        model.save(model_save_loc)
        with tf.Graph().as_default():
            with tf.Session(config=cfg).as_default():
                K.set_learning_phase(0)
                custom_objects = DIC_LOAD_MODEL.copy()
                custom_objects["custom_loss"] = custom_loss
                load_model(model_save_loc, custom_objects=custom_objects)
                session = K.get_session()
                tf.train.export_meta_graph()
                saver = tf.train.Saver()
                saver.save(session, session_save_loc)
        self.guided_graph = tf.Graph()
        with self.guided_graph.as_default():
            self.guided_sess = tf.Session(graph = self.guided_graph,config=cfg)
            with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                saver = tf.train.import_meta_graph(graph_save_loc)
                saver.restore(self.guided_sess, session_save_loc)
                self.imp_y = self.guided_graph.get_tensor_by_name(
                    model.output.name)[0][output_index]
                self.imp_x = self.guided_graph.get_tensor_by_name(model.input.name)
                self.guided_grads = tf.gradients(self.imp_y, self.imp_x)

    def get_mask(self, input_image):
        """Returns a GuidedBackprop mask."""
        x_value = np.expand_dims(input_image, axis=0)
        guided_dict = {}
        guided_dict[self.imp_x] = x_value
        gradients = self.guided_sess.run(self.guided_grads, feed_dict = guided_dict)[0][0]
        return gradients

# Yields Intersection of Area between polygons
def find_intersects(a1, a2):
    """"Finds area of intersection of two polygons """
    if  a1.intersects(a2):
        return (a1.intersection(a2)).area
    else:
        return 0

# Finds intersecting polygons
def poly_intersection(left_df, right_df, lsuffix='left', rsuffix='right'):
    index_left = 'index_%s' % lsuffix
    index_right = 'index_%s' % rsuffix
    if (any(left_df.columns.isin([index_left, index_right]))
        or any(right_df.columns.isin([index_left, index_right]))):
        raise ValueError("'{0}' and '{1}' cannot be names in the frames being"
                         " joined".format(index_left, index_right))
    left_df = left_df.copy(deep=True)
    left_df.index = left_df.index.rename(index_left)
    left_df = left_df.reset_index()
    right_df = right_df.copy(deep=True)
    right_df.index = right_df.index.rename(index_right)
    right_df = right_df.reset_index()
    # insert the bounds in the rtree spatial index
    right_df_bounds = right_df.geometry.apply(lambda x: x.bounds)
    stream = ((i, b, None) for i, b in (enumerate(right_df_bounds)))
    tree_idx = rtree.index.Index(stream)
    return (left_df, right_df,
            left_df.geometry.apply(lambda x: x.bounds) .apply(lambda x: list(tree_idx.intersection(x))))


# Spatial Join for assigning cells to SES
def sjoin(left_df, right_df, how='inner', op='intersects',
          lsuffix='left', rsuffix='right'):
    left_df, right_df, idxmatch = poly_intersection(left_df, right_df,
                                                    lsuffix='left', rsuffix='right')
    one_to_many_idxmatch = idxmatch[idxmatch.apply(len) > 0]
    if one_to_many_idxmatch.shape[0] > 0:
        r_idx = np.concatenate(one_to_many_idxmatch.values)
        l_idx = np.concatenate([[i] * len(v)
                                for i, v in one_to_many_idxmatch.iteritems()])
        check_predicates = np.vectorize(find_intersects)
        result_one_to_many = (pd.DataFrame(
            np.column_stack([l_idx, r_idx,
                             check_predicates(left_df.geometry[l_idx],
                                              right_df[right_df.geometry.name][r_idx])])))
        result_one_to_many.columns = ['_key_left', '_key_right', 'match_bool']
        result_one_to_many._key_left = result_one_to_many._key_left.astype(int)
        result_one_to_many._key_right = result_one_to_many._key_right.astype(int)
        result_one_to_many = pd.DataFrame(
            result_one_to_many[result_one_to_many['match_bool'] > 0])
        result_one_to_many = result_one_to_many.groupby("_key_right")\
                             .apply(lambda x : list(x["_key_left"]))
    return result_one_to_many

#GradCAM method for visualizing input saliency for processing multiple images in one run. (MAX_BS)
def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output,
                        np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()],
                             [layer_output, grads])
    #
    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    #
    new_cams = np.empty((images.shape[0], H, W))
    new_cams_rz = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + EPSILON) / (np.linalg.norm(cam_i, 2) + EPSILON)
        new_cams[i] =  np.maximum(cv2.resize(cam_i, (W, H), cv2.INTER_LINEAR),0)
        new_cams_rz[i] = new_cams[i] / new_cams[i].max()
    del loss, layer_output, grads, gradient_fn, conv_output, grads_val
    return new_cams, new_cams_rz

# Calculates Raster Statistics
def indiv_raster(gradcam_bg,t,test_cores,class_):
    raster_loc = OUTPUT_BASE_DIR + 'tmp/raster_gpu_{}_{}.tif'.format(city.lower(),class_)
    poly_loc = OUTPUT_BASE_DIR + 'tmp/poly_gpu_{}.shp'.format(city.lower())
    class_dataset = rasterio.open(raster_loc,'w',driver='GTiff',height=H, width=W,
                                  count=1,
                                  dtype=gradcam_bg.dtype,
                                  crs={'init': 'epsg:3035'},transform=t)
    class_dataset.write(gradcam_bg, 1)
    class_dataset.close()
    my_ops = ['sum','min','max','median','mean','count']
    test_cores[["ITEM2012","geometry"]].to_file(poly_loc)
    stats_class = zonal_stats(poly_loc,raster_loc,stats=my_ops,geojson_out=True)
    return gpd.GeoDataFrame.from_features(stats_class).rename({k:str(class_)+ "_" + k
                                                               for k in my_ops},axis=1)

# Calculates TV distance within polygons in raster
def individual_totvar(gradcam_gdf,grad_cam_poor,grad_cam_rich):
    raster_loc_poor = OUTPUT_BASE_DIR + 'tmp/raster_gpu_{}_{}.tif'.format(
        city.lower(),class_poor)
    raster_loc_rich = OUTPUT_BASE_DIR + 'tmp/raster_gpu_{}_{}.tif'.format(
        city.lower(),class_rich)
    poly_loc = OUTPUT_BASE_DIR + 'tmp/poly_gpu_{}.shp'.format(city.lower())
    if grad_cam_poor.sum() == 0 or grad_cam_rich.sum() == 0:
        tot_var_data = sym_KL = None
    else:
        tot_var_data, sym_KL = [], []
        stats_poor = zonal_stats(poly_loc,raster_loc_poor,stats='sum',raster_out=True,)
        stats_rich = zonal_stats(poly_loc,raster_loc_rich,stats='sum',raster_out=True,)
        for k in list(gradcam_gdf.index):
            P = stats_poor[k]["mini_raster_array"].data
            P /= np.sum(P[stats_poor[k]["mini_raster_array"].mask])
            P = P[stats_poor[k]["mini_raster_array"].mask]
            #
            Q = stats_rich[k]["mini_raster_array"].data
            Q /= np.sum(Q[stats_rich[k]["mini_raster_array"].mask])
            Q = Q[stats_rich[k]["mini_raster_array"].mask]
            #
            tot_var_data.append(0.5*np.sum(np.abs(P-Q)))
            sym_KL.append(entropy(P,Q)+entropy(Q,P))
    return tot_var_data, sym_KL

# Preprocess Image
def load_prepared_img(im_name):
    return cv2.resize(my_preprocessor(cv2.imread(im_name)),IMG_SIZE)

# Runs through inner batch  for 1 outer batch iteration
def serial_gradcam(model,smpl_census_imnames,indices,ind_batched_list,class_):
    # CAM
    full_grad_cams, full_grad_cam_bgs = [], []
    # Guided BackPropation
    print("Guided BackPropagation")
    guided_bprop = GuidedBackprop(model,output_index=class_);
    print("Batching")
    for i in tqdmn(range(len(ind_batched_list)-1)):
        batch_smpl_census_imnames = smpl_census_imnames[ind_batched_list[i]:ind_batched_list[i+1]]
        batch_sample_census_cell_imgs = [load_prepared_img(im)
                                         for im in batch_smpl_census_imnames]
        batch_classes = [class_
                         for j in range(ind_batched_list[i],ind_batched_list[i+1])]
        grad_cams, grad_cam_rzs = grad_cam_batch(model,
                                                 np.stack(batch_sample_census_cell_imgs),
                                                 batch_classes, conv_name)
        masks = [guided_bprop.get_mask(img)
                 for img in tqdmn(batch_sample_census_cell_imgs)]
        images = np.stack([np.sum(np.abs(mask), axis=2) for mask in tqdmn(masks)])

        # Combination
        gradcam_bgs = np.multiply(grad_cam_rzs,images)
        upper_percs = np.percentile(gradcam_bgs,99,(1,2))
        gradcam_bgs = np.minimum(gradcam_bgs,np.stack([k *np.ones((W,H))
                                                       for k in upper_percs]))
        full_grad_cams.append(grad_cams)
        full_grad_cam_bgs.append(gradcam_bgs)
    #
    full_grad_cams = np.vstack(full_grad_cams)
                     if len(full_grad_cams) > 1 else full_grad_cams
    full_grad_cam_bgs = np.vstack(full_grad_cam_bgs)
                        if  len(full_grad_cam_bgs) > 1 else full_grad_cam_bgs
    return full_grad_cams,full_grad_cam_bgs

# Calculates raster statistics for one sample
def compute_statistics(gbg_poor,gbg_rich,gcam_poor,gcam_rich,
                       t_ind,cores_ind,val_idINSPIRE,class_poor,class_rich):
    gradcam_gdf_poor = indiv_raster(gbg_poor,t_ind,cores_ind,class_poor)
    gradcam_gdf_rich = indiv_raster(gbg_rich,t_ind,cores_ind,class_rich)
    gradcam_gdf = pd.concat([gradcam_gdf_poor,
                             gradcam_gdf_rich[['4_sum','4_min','4_max',
                                               '4_median','4_mean','4_count']]],
                            axis=1)
    tot_var_data, sym_KL = individual_totvar(gradcam_gdf,gcam_poor,gcam_rich)
    gradcam_gdf["totvar"] = tot_var_data
    gradcam_gdf["sym_KL"] = sym_KL
    gradcam_gdf["area"] = gradcam_gdf.geometry.area
    gradcam_gdf["poor_score"] = gradcam_gdf["0_sum"]/gradcam_gdf["area"]
    gradcam_gdf["rich_score"] = gradcam_gdf["4_sum"]/gradcam_gdf["area"]
    return t_ind, cores_ind, gradcam_gdf, val_idINSPIRE

# Runs everything
def serialize_batch(model,ua_data,gdf_full_im_df,indices,ind_list,
                    class_poor,class_rich,ideal_workload):
    print("Overlaying")
    test_cores = [gpd.overlay(ua_data.iloc[indices[ind]],
                              gdf_full_im_df.iloc[ind:(ind+1)],
                              how='intersection') for ind in tqdmn(ind_list)]
    print("Bounding")
    ts = [from_bounds(gdf_full_im_df[ind:(ind+1)].bounds.minx.values[0]+0,
                gdf_full_im_df[ind:(ind+1)].bounds.miny.values[0]+0,
                gdf_full_im_df[ind:(ind+1)].bounds.maxx.values[0]+0,
                gdf_full_im_df[ind:(ind+1)].bounds.maxy.values[0]+0,
                W, H) for ind in tqdmn(ind_list)]
    print("Generating Images")
    sample_datas = [gdf_full_im_df.iloc[ind] for ind in tqdmn(ind_list) ]
    smpl_census_imnames = [IMG_OUTPUT_DIR + val.path2im for val in tqdmn(sample_datas)]
    #
    data = []
    for batch_idx in range(0,len(ind_list),ideal_workload):
        batch_ind_list = ind_list[batch_idx:(batch_idx+ideal_workload)]
        batch_smpl_census_imnames = smpl_census_imnames[batch_idx:(batch_idx+ideal_workload)]
        batch_indices = indices[batch_idx:(batch_idx+ideal_workload)]
        ind_batched_list = list(np.arange(0,len(batch_ind_list),MAX_BS))
        if ind_batched_list[-1] != len(batch_ind_list):
            ind_batched_list.append(len(batch_ind_list))
        #
        print("GradCaming POOR")
        gcams_poor, gbgs_poor = serial_gradcam(model,batch_smpl_census_imnames,
                                               batch_indices,ind_batched_list,
                                               class_poor)
        print("GradCaming RICH")
        gcams_rich, gbgs_rich = serial_gradcam(model,batch_smpl_census_imnames,
                                               batch_indices,ind_batched_list,
                                               class_rich)
        print("Computing raster statistics")
        for j,ind in tqdmn(enumerate(batch_ind_list)):
            val_idINSPIRE = gdf_full_im_df.iloc[ind:(ind+1)].idINSPIRE.values[0]
            data.append(
                compute_statistics(gbgs_poor[j],gbgs_rich[j],
                                   gcams_poor[j],gcams_rich[j],
                                   ts[ind],test_cores[ind],
                                   val_idINSPIRE,class_poor,class_rich))
    return data

if __name__ == '__main__':
    print("Generating Full DataSet")
    full_im_df = generate_full_idINSPIRE(UA_DIR, AERIAL_DIR, CENSUS_DIR, IMG_OUTPUT_DIR)
    city_assoc = pd.read_csv(IMG_OUTPUT_DIR + "city_assoc.csv")
    full_im_df_ua = pd.merge(full_im_df,city_assoc,on="idINSPIRE");
    full_im_df_ua = full_im_df_ua[full_im_df_ua.FUA_NAME == city]
    #
    gdf_full_im_df = full_im_df_ua.to_crs({'init': 'epsg:3035'})
    #
    print("Generating UA DataSet")
    ua_data = gpd.GeoDataFrame(pd.concat([gpd.read_file(d)
                    for d in tqdmn(glob.glob(UA_DIR+"**/Shapefiles/*UA2012.shp"))]))
    ua_data.crs = {'init': 'epsg:3035'}
    #
    print("Joining UA + Full")
    indices = sjoin(ua_data,gdf_full_im_df)
    #
    print("Loading Model")
    #indices to distribute among cores
    folds_data = pd.concat(
        [pd.read_csv(fold_file,header=0,sep=",")
         for fold_file in glob.glob(MODEL_OUTPUT_DIR+"/*last_best_models.csv")],
        axis=0).reset_index(drop=True)
    best_model_city = folds_data.ix[folds_data["Validation loss"].idxmin()]["Model file"]
    print("Loading Weights {}".format(best_model_city))
    #
    eff_model = load_model(MODEL_OUTPUT_DIR + best_model_city,
                           custom_objects=DIC_LOAD_MODEL)
    eff_model.compile(loss='categorical_crossentropy', optimizer='adam')
    #
    print("GradCaming Urban Environments")
    class_poor = 0
    class_rich = NB_SES_CLASSES - 1
    N_list = range(gdf_full_im_df.shape[0])
    test = serialize_batch(eff_model, ua_data, gdf_full_im_df,
                           indices, N_list, class_poor, class_rich, WORKLOAD)
    pickle.dump(test,
                open(OUTPUT_BASE_DIR +
                     "tmp/urbanization_patterns_{}_income.p".format(city.lower()), "wb"))
