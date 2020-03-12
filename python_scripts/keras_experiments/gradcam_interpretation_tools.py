from rasterstats import zonal_stats, point_query, gen_zonal_stats
import rasterio
import rtree
import pandas as pd
import cv2
import geopandas as gpd
import os,sys
from tqdm import tqdm as tqdmn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from skimage import io
from matplotlib.cm import inferno
import glob
from rasterio.transform import from_bounds
from functools import reduce
import pickle
from sklearn import decomposition
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
from collections import OrderedDict
from six.moves import cStringIO as StringIO
from bokeh.plotting import figure, show, output_file

# Global paths
BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/AERIAL_esa_URBAN_ATLAS_FR/"
MODEL_OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/model_data/efficientnet_keras/"
NB_SES_VALUES = 5

sns.set_style("white")
plt.style.use('bmh')

dic_rename = {
    'arable land (annual crops)': 'agri_wetland',
    'complex and mixed cultivation patterns':'agri_wetland',
    'continuous urban fabric (s.l. > 80%)': 'vhd uf' ,
    'discontinuous dense urban fabric (s.l. 50% - 80%)':'hd uf' ,
    'discontinuous low density urban fabric (s.l. 10% - 30%)':'ld uf' ,
    'discontinuous medium density urban fabric (s.l. 30% - 50%)':'md uf' ,
    'discontinuous very low density urban fabric (s.l. < 10%)':'vld uf' ,
    'fast transit roads and associated land':'motorways',
    'green urban areas': 'green ua',
    'herbaceous vegetation associations (natural grassland, moors...)': 'natural areas',
    'industrial, commercial, public, military and private units': 'comr_indst',
    'isolated structures':'isoltd_rsdnt',
    'land without current use':'no use',
    'mineral extraction and dump sites':'const/dmp',
    'construction sites': 'const/dmp',
    'orchards at the fringe of urban classes':'agri_wetland',
    'other roads and associated land':'roads',
    'pastures':'agri_wetland',
    'permanent crops (vineyards, fruit trees, olive groves)':'agri_wetland',
    'port areas':'port',
    'forests':'natural areas',
    'wetland':'agri_wetland',
    'water bodies':'water',
    'open spaces with little or no vegetation (beaches, dunes, bare rocks, glaciers)':'op_sp/beach',
    'railways and associated land':'railway',
    'sports and leisure facilities':'leis fac.',
}

UA_DATA = gpd.GeoDataFrame(pd.concat([gpd.read_file(d) for d in tqdmn(glob.glob(UA_DIR+"**/Shapefiles/*UA2012.shp"))]), 
                           crs = {'init': 'epsg:3035'})
UA_COLS = list(set(UA_DATA.ITEM2012))
treat_class = lambda x :x.lower()\
     .replace(":","")\
     .replace("  "," ")\
     .replace("…","...")\
     .replace(", ("," (")\
     .replace(", ("," (")\
     .replace("moorsâ\x80¦","moors...")\
     .replace("moors?","moors...")
format_class = lambda x : dic_rename[x] if x in dic_rename else x
final_class_fmt = lambda x : format_class(treat_class(x))
UA_COLS = set(list(map(final_class_fmt,UA_COLS)))

univariate_plot_super_color = OrderedDict([
    ("urban fabric",   "#e69584"),
    ("functional area", "sandybrown"),
    ("infrastructure",  "#aeaeb8"  ),
    ("nature",     "darkseagreen"  ),
    ("no use",     "silver"  ),
])
univariate_plot_ses_color = OrderedDict([
    ("poor_ratio_mean", "#0d3362"),
    ("rich_ratio_mean", "#c64737"),
])
unicount_plot_ses_color = OrderedDict([
    ("poor", "#0d3362"),
    ("rich", "#c64737"),
])

univariate_plot_width = 800
univariate_plot_height = 800
univariate_plot_inner_radius = 80
univariate_plot_outer_radius = 300 - 10
univariate_plot_outest_radius = 380 - 10

class_super = pd.DataFrame([
    ("agri_wetland","functional area"),
    ("comr_indst","functional area"),
    ("const/dmp","functional area"),
    ("green ua","functional area"),
    ("port","infrastructure"),
    ("hd uf","urban fabric"),
    ("isoltd_rsdnt","urban fabric"),
    ("ld uf","urban fabric"),
    ("leis fac.","functional area"),
    ("md uf","urban fabric"),
    ("motorways","infrastructure"),
    ("no use","no use"),
    ("railway","infrastructure"),
    ("roads","infrastructure"),
    ("vhd uf","urban fabric"),
    ("vld uf","urban fabric"),
    ("natural areas","nature"),
    ("water","nature"),
    ("op_sp/beach","nature"),
],columns=["ITEM2012","super_class"])

ord_index = ['agri_wetland','green ua', 'comr_indst','leis fac.','const/dmp','no use', 'roads', 'railway',
       'motorways',"port",'water',"op_sp/beach",'natural areas',  
       'isoltd_rsdnt', 'vld uf', 'ld uf', 'md uf', 'hd uf', 'vhd uf']

def univariate_pre_rad(mic):
    return np.log(mic)

def univariate_rad(mic,a,b):
    return a *pre_rad((mic))  + b

def unicount_pre_rad(mic):
    return (mic**.5)

def unicount_rad(mic,a):
    return (a*pre_rad(mic))


def pd_outlier_removal(data,cols=[],perc=95):
    """ Filters bottom and top perc/2 % outliers for provided columns """
    if len(cols) == 0:
        cols = list(data.columns)
    low_per = (100-perc)*0.5
    high_per = 100 - low_per
    if len(cols) == 1:
        val_low, val_high = np.percentile(data[cols],100-perc),np.percentile(data[cols],perc)
        truth_ser =(data[cols]<val_high)&(data[cols]>val_low)
    else:
        val_low, val_high = np.percentile(data[cols[0]],100-perc),np.percentile(data[cols[0]],perc)
        truth_ser =(data[cols[0]]<val_high)&(data[cols[0]]>val_low)
        for col in cols[1:]:
            val_low, val_high = np.percentile(data[col],100-perc),np.percentile(data[col],perc)
            truth_ser = truth_ser &((data[col]<val_high)&(data[col]>val_low))
    return data[truth_ser]

def get_attention_stats(df_level,class_idx,class_name):
    """ Get attention statistics  for predicted classes"""
    tot_attention = df_level[str(class_idx)+"_sum"].sum()
    frac_attention = df_level[str(class_idx)+"_sum"]/tot_attention
    frac_area = df_level["area"]/(df_level["area"].sum())
    pre_norm_frac_attention = frac_attention/frac_area
    norm_frac_attention = pre_norm_frac_attention/np.sum(pre_norm_frac_attention)
    score = df_level[str(class_idx)+"_sum"]/df_level["area"]
    focus_poly_score = df_level[str(class_idx)+"_sum"]/\
                        (df_level[str(class_idx)+"_max"]*df_level[str(class_idx)+"_count"])
    diffusion_expected_value = tot_attention*frac_area
    observed_value = df_level[str(class_idx)+"_sum"]
    ratio = observed_value/diffusion_expected_value
    #
    df_level[class_name + "_attention_frac"] = frac_attention
    df_level[class_name + "_attention_frac_given_area"] = norm_frac_attention
    df_level[class_name + "_score"] = score
    df_level[class_name + "_ratio"] = ratio
    return df_level

def distance_from_equilibrium(df_bivar, perc=99):
    """Computes ratio of joint mean and univariate mean and returns its distance from the equilibrium (1,1)"""
    col_a, col_b = df_bivar.columns
    df_bivar_joint = pd_outlier_removal(df_bivar.dropna(how="any"),perc=perc) #df_bivar.dropna(how="any")
    df_bivar_joint_stat = df_bivar_joint.apply([np.nanmean,np.nanstd],axis=0)
    #col_a_mean, col_b_mean = (np.nanmean(df_bivar[col_a]),np.nanmean(df_bivar[col_b]))
    col_a_mean = np.nanmean(pd_outlier_removal(df_bivar[[col_a]].dropna(how="any"),perc=perc)[col_a])
    col_b_mean = np.nanmean(pd_outlier_removal(df_bivar[[col_b]].dropna(how="any"),perc=perc)[col_b])
    stat_a = df_bivar_joint_stat[col_a].loc["nanmean"]/col_a_mean
    stat_b = df_bivar_joint_stat[col_b].loc["nanmean"]/col_b_mean
    dist_from_equilibrium = np.sqrt((stat_a-1)**2+(stat_b-1)**2)
    return (stat_a, stat_b, dist_from_equilibrium)

def bootstrapped_distance_from_equilibrium(df_ua_data_ratio, col_a, col_b,perc_smpls,nb_iter,low_ic, high_ic):
    """ Returns observed, IC95% and median value for the distance_from_equilibrium metric for pair of columns"""
    dataset = df_ua_data_ratio[[col_a,col_b]]
    (stat_a, stat_b, dist) = distance_from_equilibrium(dataset)
    boot_data=[distance_from_equilibrium(dataset.sample(frac=perc_smpls)) for _ in range(nb_iter)]
    df_boot_data = pd.DataFrame(boot_data,columns=["stat_a","stat_b","dist"])
    stat_a_boot,stat_b_boot,stat_dist_boot = list(map(np.sort,df_boot_data.T.values))
    return (stat_a,stat_a_boot[low_ic],np.median(stat_a_boot),stat_a_boot[high_ic],
            stat_b,stat_b_boot[low_ic],np.median(stat_b_boot),stat_b_boot[high_ic],
            dist,stat_dist_boot[low_ic],np.median(stat_dist_boot),stat_dist_boot[high_ic])

def quadrant_bootstrapped_coactivations(df_ua_data_ratio, min_nb_pts, ic_alpha=.05, nb_iter = 100, perc_smpls = .8):
    """ 
    Generates coactivation statistics 
        - ic_alpha: percentage confidence interval statistics
        - min_nb_pts: minimal # of observed pairs needed to compute coactivations
        - nb_iter: # Iterations of the bootstrap algorithm to generate stats
        - perc_smpls: Percentage of samples used in computing bootstrapped statistics 
    """
    #df_ua_data_ratio.columns = [dic_rename[k] if k in dic_rename else k for k in df_ua_data_ratio.columns]
    cols = list(df_ua_data_ratio.columns)
    col_labs = {col:str(i) for i,col in enumerate(cols)}
    low_ic, high_ic = int((ic_alpha/2.0)*100), int((1-ic_alpha/2.0)*100)
    col_pairs_to_treat = [(col_a,col_b)
                          for i,col_a in enumerate(cols[:-1]) for j,col_b in enumerate(cols[(i+1):])
                          if df_ua_data_ratio[[col_a,col_b]].dropna(how="any").shape[0]*perc_smpls > min_nb_pts]
    full_data = Parallel(n_jobs=10)(
        delayed(bootstrapped_distance_from_equilibrium)
        (df_ua_data_ratio,col_a, col_b,perc_smpls,nb_iter,low_ic, high_ic)
        for col_a, col_b in col_pairs_to_treat)
    #
    full_stats = [("{}-{}".format(col_a,col_b),)+stats for (col_a, col_b), stats in zip(col_pairs_to_treat,full_data)]
    X = np.array([stats[0] for stats in full_data])
    Y = np.array([stats[4] for stats in full_data])
    full_stats_df = pd.DataFrame(
        full_stats,columns=["pair",]+[h+k for k in ["_a","_b","_dist"]for h in ["emp","low_ic","med","high_ic"]])
    data_eff = pd.DataFrame([("{}-{}".format(col_a,col_b),df_ua_data_ratio[[col_a,col_b]].dropna(how="any").shape[0])
                for (col_a, col_b) in col_pairs_to_treat],columns=["pair","nb_samples"])
    full_stats_df = pd.merge(full_stats_df,data_eff,on="pair")
    return X, Y, full_stats_df.set_index("pair")

def extract_data_from_gradcam_dic_with_per(d_res, idINSPIRE_attention_df, per_poor = 0, per_rich = 0):
    """ Extract data from gradcam comps that are above the top (per_poor, per_rich) activation percentiles """
    rough_score_tile = {"poor":[],"rich":[]}
    ua_data_ratio_tile = {"poor":[],"rich":[]}
    for i, (_, ua_df, df_poly, idINSPIRE) in tqdmn(enumerate(d_res)):
        df_poly["ITEM2012"] = df_poly.ITEM2012.apply(final_class_fmt)
        df_tile_sums = df_poly.groupby("ITEM2012")[["0_sum","4_sum","0_count","4_count","area"]].sum().reset_index()
        df_tile_maxs = df_poly.groupby("ITEM2012")[["0_max","4_max"]].max().reset_index()
        df_tile = pd.concat([df_tile_sums,df_tile_maxs.drop("ITEM2012",axis=1)],axis=1)
        class_indices, class_labels = [], []       
        curr_row = idINSPIRE_attention_df[idINSPIRE_attention_df.idINSPIRE == idINSPIRE]
        if curr_row.poor_percentile.values[0] > per_poor:
            class_indices.append(0)
            class_labels.append("poor")
        if curr_row.rich_percentile.values[0] > per_rich:
            class_indices.append(NB_SES_VALUES -1)
            class_labels.append("rich")
        for class_idx,class_name in zip(class_indices,class_labels):
            #df_level = get_attention_stats(df_poly,class_idx,class_name)
            df_level_tile = get_attention_stats(df_tile,class_idx,class_name)
            dic_cnt_ratios = {k:[np.nan] for k in UA_COLS}
            for class_ua,val_scores_ua in zip(df_level_tile.ITEM2012,df_level_tile[class_name+"_ratio"]):
                dic_cnt_ratios[class_ua] = [val_scores_ua]
            ua_data_ratio_tile[class_name].append(pd.DataFrame.from_dict(dic_cnt_ratios))
            rough_score_tile[class_name].append(df_level_tile)
    df_rough_poor_score_tile = pd.concat(rough_score_tile["poor"],axis=0).reset_index(drop=True)
    df_rough_rich_score_tile = pd.concat(rough_score_tile["rich"],axis=0).reset_index(drop=True)
    df_ua_data_ratio_poor_tile = pd.concat(ua_data_ratio_tile["poor"], axis=0)
    df_ua_data_ratio_rich_tile = pd.concat(ua_data_ratio_tile["rich"], axis=0)
    return (df_rough_poor_score_tile,df_rough_rich_score_tile,df_ua_data_ratio_poor_tile, df_ua_data_ratio_rich_tile)

def extract_data_from_gradcam_dic_if_class(d_res, idINSPIRE_attention_df, class_poor = [0,1], class_rich = [3,4]):
    """ Extract data from gradcam comps that are in classes """
    rough_score_tile = {"poor":[],"rich":[]}
    ua_data_ratio_tile = {"poor":[],"rich":[]}
    for i, (_, ua_df, df_poly, idINSPIRE) in tqdmn(enumerate(d_res)):
        df_poly["ITEM2012"] = df_poly.ITEM2012.apply(final_class_fmt)
        df_tile_sums = df_poly.groupby("ITEM2012")[["0_sum","4_sum","0_count","4_count","area"]].sum().reset_index()
        df_tile_maxs = df_poly.groupby("ITEM2012")[["0_max","4_max"]].max().reset_index()
        df_tile = pd.concat([df_tile_sums,df_tile_maxs.drop("ITEM2012",axis=1)],axis=1)
        class_indices, class_labels = [], []       
        curr_row = idINSPIRE_attention_df[idINSPIRE_attention_df.idINSPIRE == idINSPIRE]
        if curr_row.out.values[0] in class_poor:
            class_indices.append(0)
            class_labels.append("poor")
        elif curr_row.out.values[0] in class_rich:
            class_indices.append(NB_SES_VALUES -1)
            class_labels.append("rich")
        for class_idx,class_name in zip(class_indices,class_labels):
            #df_level = get_attention_stats(df_poly,class_idx,class_name)
            df_level_tile = get_attention_stats(df_tile,class_idx,class_name)
            dic_cnt_ratios = {k:[np.nan] for k in UA_COLS}
            for class_ua,val_scores_ua in zip(df_level_tile.ITEM2012,df_level_tile[class_name+"_ratio"]):
                dic_cnt_ratios[class_ua] = [val_scores_ua]
            ua_data_ratio_tile[class_name].append(pd.DataFrame.from_dict(dic_cnt_ratios))
            rough_score_tile[class_name].append(df_level_tile)
    df_rough_poor_score_tile = pd.concat(rough_score_tile["poor"],axis=0).reset_index(drop=True)
    df_rough_rich_score_tile = pd.concat(rough_score_tile["rich"],axis=0).reset_index(drop=True)
    df_ua_data_ratio_poor_tile = pd.concat(ua_data_ratio_tile["poor"], axis=0)
    df_ua_data_ratio_rich_tile = pd.concat(ua_data_ratio_tile["rich"], axis=0)
    return (df_rough_poor_score_tile,df_rough_rich_score_tile,df_ua_data_ratio_poor_tile, df_ua_data_ratio_rich_tile)


def organize_univariate_results(df_score_poly, ses_class):
    """ Generates stats on univariate data for radarplot mean +/- std"""
    uni_ua_mean= df_score_poly.groupby("ITEM2012")[["{}_ratio".format(ses_class)]].mean().reset_index()
    uni_ua_std = df_score_poly.groupby("ITEM2012")[["{}_ratio".format(ses_class)]].std().reset_index()
    uni_ua_cnt = df_score_poly.groupby("ITEM2012")[["{}_ratio".format(ses_class)]].count().reset_index()
    uni_ua_full = pd.concat([uni_ua_mean,
                             uni_ua_std.drop("ITEM2012",axis=1),
                             uni_ua_cnt.drop("ITEM2012",axis=1)],axis=1)
    uni_ua_full.columns = ["ITEM2012"] + ["{}_ratio_{}".format(ses_class,col) for col in ["mean","std","cnt"]]
    est_mean= uni_ua_full["{}_ratio_mean".format(ses_class)]
    est_stderr = uni_ua_full["{}_ratio_std".format(ses_class)]/np.sqrt(uni_ua_full["{}_ratio_cnt".format(ses_class)])
    uni_ua_full["{}_ratio_stderr".format(ses_class)] = est_stderr
    return uni_ua_full

def organize_bivariate_results(df_ratio_ses_poly, min_nb_points):
    """ Generates stats on bivariate data for barplot"""
    #cols_of_interest = ["low_ic_dist", "med_dist", "high_ic_dist","nb_samples"]
    (_, _, df_stats_ses) = quadrant_bootstrapped_coactivations(df_ratio_ses_poly, min_nb_points)

    #ses_coexcitation = df_stats_ses[(df_stats_ses.emp_a>1)&(df_stats_ses.emp_b>1)][cols_of_interest].reset_index()
    ses_coactivation = df_stats_ses.reset_index()
    ses_coexcitation = ses_coactivation[(ses_coactivation.emp_a>1)&(ses_coactivation.emp_b>1)]
    ses_coinhibition = ses_coactivation[(ses_coactivation.emp_a<1)&(ses_coactivation.emp_b<1)]
    #ses_coinhibition = df_stats_ses[(df_stats_ses.emp_a<1)&(df_stats_ses.emp_b<1)][cols_of_interest].reset_index()
    return ses_coactivation,ses_coexcitation, ses_coinhibition

def get_plot_data(test_summary,min_nb_points = 50):
    """ Returns data to plot from treated gradcam results"""
    (df_poor_score_poly,df_rich_score_poly,df_ratio_poor_poly, df_ratio_rich_poly) = test_summary
    uni_ua_poor = organize_univariate_results(df_poor_score_poly, "poor")
    uni_ua_rich = organize_univariate_results(df_rich_score_poly, "rich")
    poor_coact, poor_coexcitation, poor_coinhibition =  organize_bivariate_results(df_ratio_poor_poly, min_nb_points)
    rich_coact, rich_coexcitation, rich_coinhibition =  organize_bivariate_results(df_ratio_rich_poly, min_nb_points)
    return (uni_ua_poor, poor_coact, poor_coexcitation, poor_coinhibition,
            uni_ua_rich, rich_coact, rich_coexcitation, rich_coinhibition)