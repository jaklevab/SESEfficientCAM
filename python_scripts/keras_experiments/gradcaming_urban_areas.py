from rasterstats import zonal_stats, point_query, gen_zonal_stats
import keras
import rasterio
import rtree
from keras import backend as K
from keras.optimizers import SGD, Adam
import pandas as pd
import cv2
import geopandas as gpd
import os,sys
from tqdm import tqdm as tqdmn
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras import metrics
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Input
from skimage import io
from matplotlib.cm import inferno
import glob
from rasterio.transform import from_bounds
from efficientnet import EfficientNetB0 as EfficientNet
sys.path.append("/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/satellite/")
from generate_fr_ua_vhr_data import generate_car_census_data
sys.path.append("/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/aerial/")
from aerial_training_utils import generate_full_idINSPIRE, geographical_boundaries, my_preprocessor, fmeasure,recall,precision, fbeta_score
import tensorflow as tf
from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
from functools import reduce
from scipy.stats import entropy
import pickle
from keras import backend as K

# Global paths
BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/AERIAL_esa_URBAN_ATLAS_FR/"
MODEL_OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/model_data/efficientnet_keras/"

# Global variables
W = H = 800
IMG_SIZE = (W, H)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
NB_SES_CLASSES = 5
ADRIAN_ALBERT_THRESHOLD = .25
INSEE_AREA = 200*200
NB_SAMPLED = 5000
MAX_BS = 10 #(grunch) 35 (rrunch) 

conv_name = 'conv2d_65'
input_name = 'input_1'

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

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

class GuidedBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with GuidedBackProp.
    This implementation copies the TensorFlow graph to a new graph with the ReLU
    gradient overwritten as in the paper:
    https://arxiv.org/abs/1412.6806
    """
    GuidedReluRegistered = False
    def __init__(self, model, output_index=0, custom_loss=None):
        """Constructs a GuidedBackprop SaliencyMask."""

        if GuidedBackprop.GuidedReluRegistered is False:
            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0, "float32")
                gate_y = tf.cast(op.outputs[0] > 0, "float32")
                return gate_y * gate_g * grad
        GuidedBackprop.GuidedReluRegistered = True
        """ 
            Create a dummy session to set the learning phase to 0 (test mode in keras) without 
            inteferring with the session in the original keras model. This is a workaround
            for the problem that tf.gradients returns error with keras models that contains 
            Dropout or BatchNormalization.

            Basic Idea: save keras model => create new keras model with learning phase set to 0 => save
            the tensorflow graph => create new tensorflow graph with ReLU replaced by GuiededReLU.
        """   
        model.save('/tmp/gb_keras.h5') 
        with tf.Graph().as_default(): 
            with tf.Session(config=cfg).as_default(): 
                K.set_learning_phase(0)
                load_model('/tmp/gb_keras.h5', custom_objects={"custom_loss":custom_loss})
                session = K.get_session()
                tf.train.export_meta_graph()
                saver = tf.train.Saver()
                saver.save(session, '/tmp/guided_backprop_ckpt')
        self.guided_graph = tf.Graph()
        with self.guided_graph.as_default():
            self.guided_sess = tf.Session(graph = self.guided_graph,config=cfg)
            with self.guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
                saver = tf.train.import_meta_graph('/tmp/guided_backprop_ckpt.meta')
                saver.restore(self.guided_sess, '/tmp/guided_backprop_ckpt')
                self.imported_y = self.guided_graph.get_tensor_by_name(model.output.name)[0][output_index]
                self.imported_x = self.guided_graph.get_tensor_by_name(model.input.name)
                self.guided_grads_node = tf.gradients(self.imported_y, self.imported_x)

    def get_mask(self, input_image):
        """Returns a GuidedBackprop mask."""
        x_value = np.expand_dims(input_image, axis=0)
        guided_feed_dict = {}
        guided_feed_dict[self.imported_x] = x_value
        gradients = self.guided_sess.run(self.guided_grads_node, feed_dict = guided_feed_dict)[0][0]
        return gradients

    def get_mult_mask(self, input_images):
        """Returns a GuidedBackprop mask."""
        guided_feed_dict = {}
        guided_feed_dict[self.imported_x] = input_images
        gradients = self.guided_sess.run(self.guided_grads_node, feed_dict = guided_feed_dict)[0][0]
        return gradients

def find_intersects(a1, a2):
    if  a1.intersects(a2):
        return (a1.intersection(a2)).area
    else:
        return 0

def sjoin(left_df, right_df, how='inner', op='intersects', lsuffix='left', rsuffix='right'):
    index_left = 'index_%s' % lsuffix
    index_right = 'index_%s' % rsuffix
    if (any(left_df.columns.isin([index_left, index_right]))
            or any(right_df.columns.isin([index_left, index_right]))):
        raise ValueError("'{0}' and '{1}' cannot be names in the frames being"
                         " joined".format(index_left, index_right))
    #
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
    idxmatch = (left_df.geometry.apply(lambda x: x.bounds)
                .apply(lambda x: list(tree_idx.intersection(x))))
    #
    one_to_many_idxmatch = idxmatch[idxmatch.apply(len) > 0]
    if one_to_many_idxmatch.shape[0] > 0:
        r_idx = np.concatenate(one_to_many_idxmatch.values)
        l_idx = np.concatenate([[i] * len(v) for i, v in one_to_many_idxmatch.iteritems()])
        check_predicates = np.vectorize(find_intersects)
        result_one_to_many = (pd.DataFrame(np.column_stack([l_idx, r_idx,
                                                            check_predicates(left_df.geometry[l_idx],
                                                            right_df[right_df.geometry.name][r_idx])])))
        result_one_to_many.columns = ['_key_left', '_key_right', 'match_bool']
        result_one_to_many._key_left = result_one_to_many._key_left.astype(int)
        result_one_to_many._key_right = result_one_to_many._key_right.astype(int)
        result_one_to_many = pd.DataFrame(result_one_to_many[result_one_to_many['match_bool'] > 0])
        result_one_to_many = result_one_to_many.groupby("_key_right").apply(lambda x : list(x["_key_left"]))
    return result_one_to_many

def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])
    #
    conv_output, grads_val = gradient_fn([images, 0])    
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)
    #
    # Process CAMs
    new_cams = np.empty((images.shape[0], H, W))
    new_cams_rz = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (W, H), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams_rz[i] = new_cams[i] / new_cams[i].max()  
    del loss, layer_output, grads, gradient_fn, conv_output, grads_val
    return new_cams, new_cams_rz

def individual_rastering(gradcam_bg,t,test_cores,class_):
    class_dataset = rasterio.open(BASE_DIR + 'tmp/new_test_%s.tif'%str(class_),'w',driver='GTiff',
                                height=H, width=W,count=1,dtype=gradcam_bg.dtype,
                                  crs={'init': 'epsg:3035'},transform=t)
    class_dataset.write(gradcam_bg, 1)
    class_dataset.close()
    my_ops = ['sum','min','max','median','mean','count']
    test_cores[["ITEM2012","geometry"]].to_file(BASE_DIR + 'tmp/new_poly2.shp')
    stats_class = zonal_stats(BASE_DIR + 'tmp/new_poly2.shp',BASE_DIR + 'tmp/new_test_%s.tif'%str(class_),
                              stats=my_ops,geojson_out=True)
    return gpd.GeoDataFrame.from_features(stats_class).rename({k:str(class_)+ "_" + k for k in my_ops},axis=1)

def individual_totvar(gradcam_gdf,grad_cam_poor,grad_cam_rich):
    if grad_cam_poor.sum() == 0 or grad_cam_rich.sum() == 0:
        tot_var_data = sym_KL = None
    else:
        tot_var_data, sym_KL = [], []
        stats_totvar_poor = zonal_stats(
            BASE_DIR + 'tmp/new_poly2.shp',BASE_DIR + 'tmp/new_test_%s.tif'%str(class_poor),
            stats='sum',raster_out=True,)
        stats_totvar_rich = zonal_stats(
            BASE_DIR + 'tmp/new_poly2.shp',BASE_DIR + 'tmp/new_test_%s.tif'%str(class_rich),
            stats='sum',raster_out=True,)
        for k in list(gradcam_gdf.index):
            P = stats_totvar_poor[k]["mini_raster_array"].data
            P /= np.sum(P[stats_totvar_poor[k]["mini_raster_array"].mask])
            P = P[stats_totvar_poor[k]["mini_raster_array"].mask]
            #
            Q = stats_totvar_rich[k]["mini_raster_array"].data
            Q /= np.sum(Q[stats_totvar_rich[k]["mini_raster_array"].mask])
            Q = Q[stats_totvar_rich[k]["mini_raster_array"].mask]
            #
            tot_var_data.append(0.5*np.sum(np.abs(P-Q)))
            sym_KL.append(entropy(P,Q)+entropy(Q,P))
    return tot_var_data, sym_KL

def serialize_gradcaming(model,sample_census_cell_imgs,indices,ind_batched_list,class_):
    # CAM
    grad_cams, grad_cam_rzs = [], []
    print("Batching")
    for i in tqdmn(range(len(ind_batched_list)-1)):
        batch_sample_census_cell_imgs = sample_census_cell_imgs[ind_batched_list[i]:ind_batched_list[i+1]]
        batch_classes = [class_ for j in range(ind_batched_list[i],ind_batched_list[i+1])]
        pre_grad_cams, pre_grad_cam_rzs = grad_cam_batch(model,np.stack(batch_sample_census_cell_imgs),
                                                         batch_classes, conv_name)
        grad_cams.append(pre_grad_cams)
        grad_cam_rzs.append(pre_grad_cam_rzs)
    grad_cams = np.vstack(grad_cams) if len(grad_cams) > 1 else pre_grad_cams
    grad_cam_rzs = np.vstack(grad_cam_rzs) if  len(grad_cam_rzs) > 1 else pre_grad_cam_rzs
    #
    # Guided BackPropation  
    print("Loading GB Model")
    guided_bprop = GuidedBackprop(model,output_index=class_);
    print("Guided BackPropagation")
    masks = [guided_bprop.get_mask(img) for img in tqdmn(sample_census_cell_imgs)]
    images = np.stack([np.sum(np.abs(mask), axis=2) for mask in tqdmn(masks)])
    #
    # Combination
    gradcam_bgs = np.multiply(grad_cam_rzs,images)
    upper_percs = np.percentile(gradcam_bgs,99,(1,2))
    gradcam_bgs = np.minimum(gradcam_bgs,np.stack([k *np.ones((W,H)) for k in upper_percs]))
    return grad_cams,gradcam_bgs
    
def serialize_treating(model,ua_data,gdf_full_im_df_sampled,indices,ind_list,class_poor,class_rich):
    print("Overlaying")
    test_cores = [gpd.overlay(ua_data.iloc[indices[ind]],
                             gdf_full_im_df_sampled.iloc[ind:(ind+1)], how='intersection')
                 for ind in tqdmn(ind_list)]
    print("Bounding")
    ts = [from_bounds(gdf_full_im_df_sampled[ind:(ind+1)].bounds.minx.values[0]+0,
                gdf_full_im_df_sampled[ind:(ind+1)].bounds.miny.values[0]+0,
                gdf_full_im_df_sampled[ind:(ind+1)].bounds.maxx.values[0]+0,
                gdf_full_im_df_sampled[ind:(ind+1)].bounds.maxy.values[0]+0,
                W, H)
          for ind in tqdmn(ind_list)]
    print("Generating Images")
    sample_cell_datas = [gdf_full_im_df_sampled.iloc[ind] for ind in tqdmn(ind_list) ]
    sample_census_cell_imnames = [OUTPUT_DIR + val.path2im for val in tqdmn(sample_cell_datas)]
    sample_census_cell_imgs = [cv2.resize(my_preprocessor(cv2.imread(im_name)),IMG_SIZE)
                            for im_name in tqdmn(sample_census_cell_imnames)]
    ind_batched_list = list(np.arange(0,len(ind_list),MAX_BS))
    if ind_batched_list[-1] != len(ind_list):
        ind_batched_list.append(len(ind_list))
    #
    print("GradCaming the sparse")
    gradcam_cams_poor, gradcam_bgs_poor = serialize_gradcaming(model,sample_census_cell_imgs,indices,ind_batched_list,class_poor)
    print("GradCaming the dense")
    gradcam_cams_rich, gradcam_bgs_rich = serialize_gradcaming(model,sample_census_cell_imgs,indices,ind_batched_list,class_rich)
    data = []
    print("Computing raster statistics")
    for ind in tqdmn(ind_list):
        gradcam_gdf_poor = individual_rastering(gradcam_bgs_poor[ind],ts[ind],test_cores[ind],class_poor)
        gradcam_gdf_rich = individual_rastering(gradcam_bgs_rich[ind],ts[ind],test_cores[ind],class_rich)
        gradcam_gdf = pd.concat([gradcam_gdf_poor,
                                 gradcam_gdf_rich[['4_sum','4_min','4_max','4_median','4_mean','4_count']]],axis=1)
        tot_var_data, sym_KL = individual_totvar(gradcam_gdf,gradcam_cams_poor[ind],gradcam_cams_rich[ind])
        gradcam_gdf["totvar"] = tot_var_data
        gradcam_gdf["sym_KL"] = sym_KL
        gradcam_gdf["area"] = gradcam_gdf.geometry.area
        gradcam_gdf["poor_score"] = gradcam_gdf["0_sum"]/gradcam_gdf["area"]
        gradcam_gdf["rich_score"] = gradcam_gdf["4_sum"]/gradcam_gdf["area"]
        val_idINSPIRE = gdf_full_im_df_sampled.iloc[ind:(ind+1)].idINSPIRE.values[0]
        data.append((gradcam_bgs_poor[ind],gradcam_bgs_rich[ind],ts[ind],test_cores[ind],gradcam_gdf,val_idINSPIRE))
    return data


if __name__ == '__main__':
    print("Generating Full DataSet")
    full_im_df = generate_full_idINSPIRE(UA_DIR, OUTPUT_DIR, NB_SES_CLASSES, ADRIAN_ALBERT_THRESHOLD, INSEE_AREA)
    full_im_df_sampled = full_im_df.sample(NB_SAMPLED)
    gdf_full_im_df_sampled = full_im_df_sampled.to_crs({'init': 'epsg:3035'})
    #
    print("Generating UA DataSet")
    ua_data = gpd.GeoDataFrame(pd.concat([gpd.read_file(d) 
                                          for d in tqdmn(glob.glob(UA_DIR+"**/Shapefiles/*UA2012.shp"))]))
    ua_data.crs = {'init': 'epsg:3035'}
    #
    print("Joining UA + Full")
    indices = sjoin(ua_data,gdf_full_im_df_sampled)
    #
    # Load the last best model
    dic_load_model = {
        "precision":precision,
        "recall":recall,
        "fbeta_score":fbeta_score,
        "fmeasure":fmeasure
    }
    #
    print("Loading Model")
    efficientnet_model = load_model(
        MODEL_OUTPUT_DIR + "../efficientnet_keras/density/" + "lastbest-0.hdf5",custom_objects=dic_load_model)
    efficientnet_model.compile(loss='categorical_crossentropy', optimizer='adam')
    #
    print("GradCaming Urban Environments")
    class_poor = 0
    class_rich = NB_SES_CLASSES - 1
    test = serialize_treating(efficientnet_model,ua_data,gdf_full_im_df_sampled,indices,range(NB_SAMPLED),class_poor,class_rich)
    pickle.dump(test, open(MODEL_OUTPUT_DIR + "density/urbanization_patterns.p", "wb"))
