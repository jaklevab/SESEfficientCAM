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
from scipy.stats import entropy
import pickle
from tqdm import tqdm as tqdmn
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib.gridspec as gridspec
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

from collections import OrderedDict

import numpy as np
import pandas as pd
from six.moves import cStringIO as StringIO
from bokeh.plotting import figure, show, output_file

# Global paths
import sys
import scipy.stats as st

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

ord_ITEM2012 = ['isoltd_rsdnt','vld uf','ld uf','md uf','hd uf','vhd uf',
                'roads','motorways','railway', 'no use',
                'leis fac.','comr_indst','const/dmp',
                'agri_wetland','natural areas', 'green ua',
                'water','port', 'op_sp/beach',
               ]
income_vals = pd.read_csv(OUTPUT_DIR+"2019_income_norm.csv")
filter_min = lambda x: x[(x.nb_samples>nb_min)&\
                         ([("uf" in y) and ("roads" not in y) for y in x.pair])]
stand_pair_in = lambda x : "-".join(np.sort(x.lower().split("-")).tolist())
stand_pair = lambda x : [stand_pair_in(y) for y in x]

sys.path.append(BASE_DIR + "SATELSES/equirect_proj_test/cnes/python_files/keras_experiments/efficientnet/")
from gradcam_interpretation_tools import *
from tqdm import tqdm as tqdmn

def stress_test_coactivations(df_arr,cut_arr):
    for df,cut in zip(df_arr,cut_arr):
        df["pair"] = stand_pair(df["pair"])
        df["cut"] = cut
    return  pd.concat([df.set_index(["pair","cut"]) for df in df_arr])

def generate_stats_emp(dic_results):
    data_count_univar = []
    data_count_bivar = []
    ua_cols = list(UA_COLS)
    for d in tqdmn(dic_results):
        dic_cnt_ratios_uni = {k.lower():[0] for k in UA_COLS}
        dic_cnt_ratios_bivar = {stand_pair_in("{}-{}".format(k1.lower(),k2.lower())):[0]
                                for i,k1 in enumerate(ua_cols) for k2 in ua_cols[(i+1):]}
        df_silly = d[2].groupby("ITEM2012")[["0_count"]].count().reset_index()
        for i,(class_ua1,cnt_val )in enumerate(zip(df_silly.ITEM2012,df_silly["0_count"])):
            dic_cnt_ratios_uni[class_ua1] = [cnt_val]
            class_ua1 = class_ua1.lower()
            for j,class_ua2 in enumerate(df_silly.iloc[(i+1):].ITEM2012):
                class_ua2 = class_ua2.lower()
                dic_cnt_ratios_bivar[stand_pair_in("{}-{}".format(class_ua1,class_ua2))] = [1]
        dic_cnt_ratios_uni["idINSPIRE"] = [d[3]]
        dic_cnt_ratios_bivar["idINSPIRE"] = [d[3]]
        data_count_univar.append(pd.DataFrame.from_dict(dic_cnt_ratios_uni))
        data_count_bivar.append(pd.DataFrame.from_dict(dic_cnt_ratios_bivar))
    df_uni_count = pd.concat(data_count_univar,axis=0).reset_index(drop=True)
    df_bi_count = pd.concat(data_count_bivar,axis=0).reset_index(drop=True)
    return df_uni_count, df_bi_count

def generate_univar_emp(df_uni_count):
    ua_uni_income = pd.merge(df_uni_count,income_vals,on="idINSPIRE").drop(["FUA_NAME"],axis=1)
    ua_uni_income_poor = (ua_uni_income[(ua_uni_income.treated_citywise_income==0) | (ua_uni_income.treated_citywise_income==1)].set_index("idINSPIRE").drop(
        "treated_citywise_income",axis=1)>0).astype(int).sum()
    ua_uni_income_rich = (ua_uni_income[(ua_uni_income.treated_citywise_income==3) | (ua_uni_income.treated_citywise_income==4)].set_index("idINSPIRE").drop(
        "treated_citywise_income",axis=1)>0).astype(int).sum()
    ua_uni_income_tot = (ua_uni_income.set_index("idINSPIRE").drop(
        "treated_citywise_income",axis=1)>0).astype(int).sum()
    ua_uni_income_poor_frac = (ua_uni_income_poor/ua_uni_income_tot).reset_index()
    ua_uni_income_poor_frac.columns = ["ITEM2012","poor"]
    ua_uni_income_rich_frac = (ua_uni_income_rich/ua_uni_income_tot).reset_index()
    ua_uni_income_rich_frac.columns = ["ITEM2012","rich"]
    final_uni_cnt = pd.concat([ua_uni_income_poor_frac.set_index("ITEM2012"),
                               ua_uni_income_rich_frac.set_index("ITEM2012")],axis=1)
    return final_uni_cnt

def generate_bivar_emp(df_bi_count):
    ua_bi_income = pd.merge(df_bi_count,income_vals,on="idINSPIRE").drop(["FUA_NAME"],axis=1)
    ua_bi_income_poor = (ua_bi_income[(ua_bi_income.treated_citywise_income==0) | (ua_bi_income.treated_citywise_income==1)].set_index("idINSPIRE").drop(
        "treated_citywise_income",axis=1)>0).astype(int).sum()
    ua_bi_income_rich = (ua_bi_income[(ua_bi_income.treated_citywise_income==3) | (ua_bi_income.treated_citywise_income==4)].set_index("idINSPIRE").drop(
        "treated_citywise_income",axis=1)>0).astype(int).sum()
    ua_bi_income_tot = (ua_bi_income.set_index("idINSPIRE").drop(
        "treated_citywise_income",axis=1)>0).astype(int).sum()
    ua_bi_income_poor_frac = (ua_bi_income_poor/ua_bi_income_tot).reset_index()
    ua_bi_income_poor_frac.columns = ["pair","poor"]
    ua_bi_income_rich_frac = (ua_bi_income_rich/ua_bi_income_tot).reset_index()
    ua_bi_income_rich_frac.columns = ["pair","rich"]
    final_bi_cnt = pd.concat([ua_bi_income_poor_frac.set_index("pair"),ua_bi_income_rich_frac.set_index("pair")],axis=1)
    return final_bi_cnt

def get_full_coactivation_emp(final_uni_cnt,final_bi_cnt):
    data = {"poor":[],"rich":[]}
    for col_pair in tqdmn(final_bi_cnt.index):
        col1,col2 = col_pair.split("-")
        for ses_class in ["poor","rich"]:
            #p(SES=4|UA1->UA2)=p(SES=4|UA2->UA1)
            p1_ses=(final_bi_cnt[ses_class][col_pair])
            #p(SES=4|UA1)
            p2x_ses=final_uni_cnt[ses_class][col1]
            #p(SES=4|UA2)
            p2y_ses=final_uni_cnt[ses_class][col2]
            # How much more likely is each class of being rich than without being in the vicinity of each other
            r1, r2 = (p1_ses/p2x_ses,p1_ses/p2y_ses) 
            data[ses_class].append((col1,col2,r1,r2,np.sqrt((r1-1)**2+(r2-1)**2)))
    for ses_class in ["poor","rich"]:
        data[ses_class] = pd.DataFrame(data[ses_class],columns=["ua1","ua2","r1","r2","dist"])
    return data

def bootstrapped_generate_univar(df_uni_count,nb_iter = 100,perc_smpls = .8,ic_alpha=.05):
    low_ic_val, high_ic_val = int((ic_alpha/2.0)*100), int((1-ic_alpha/2.0)*100)
    boot_data=[generate_univar_emp(df_uni_count.sample(frac=perc_smpls)).reset_index() for _ in range(nb_iter)]
    #
    def low_ic(x):
        return np.nanpercentile(x,low_ic_val)
    #
    def high_ic(x):
        return np.nanpercentile(x,high_ic_val)
    #
    df_boot_table = pd.pivot_table(pd.concat(boot_data,axis=0,ignore_index=True),
                    index=['ITEM2012',],values=['poor','rich'], aggfunc=[low_ic,np.nanmedian,high_ic],
                    fill_value=np.nan)
    #
    cols = list(df_boot_table.index)
    df_boot_data_poor=pd.DataFrame([(col,)+tuple([df_boot_table.loc[col,f]["poor"] for f in ["low_ic","nanmedian","high_ic"]])
                         for col in cols],columns=["ITEM2012","poor_low_ic","poor_nanmedian","poor_high_ic"])
    df_boot_data_rich=pd.DataFrame([tuple([df_boot_table.loc[col,f]["rich"]for f in ["low_ic","nanmedian","high_ic"]])
                         for col in cols],columns=["rich_low_ic","rich_nanmedian","rich_high_ic"])
    return pd.concat([df_boot_data_poor,df_boot_data_rich],axis=1)

def extract_valuables_from_city(city):
    city_dir = MODEL_OUTPUT_DIR+"2019_income_norm_v2/{}/preds/".format(city)
    d_res_city = pickle.load(open(city_dir+"urbanization_patterns_cpu_{}_income.p".format(city),"rb"))
    pred_data_city = pd.read_csv(city_dir + "/full_whole_predicted_values.csv",header=0)
    pred_val_cols = ["pred_val_0","pred_val_1", "pred_val_2","pred_val_3","pred_val_4"]
    pred_data_city["out"] = np.argmax(pred_data_city[pred_val_cols].values,axis=1)
    
    #GradCAM Activation univariate, bivariate stats
    test_summary_city = extract_data_from_gradcam_dic_if_class(d_res_city, pred_data_city,
                                                          class_poor = [0,1], class_rich = [3,4])

    (uni_ua_poor_city,poor_coact_city,poor_coex_city,poor_coinh_city,
     uni_ua_rich_city,rich_coact_city,rich_coex_city, rich_coinh_city)= get_plot_data(test_summary_city, min_nb_points=5)

    reord_uni_ua_poor_city = uni_ua_poor_city[uni_ua_poor_city.ITEM2012!="airports"].set_index("ITEM2012"\
                                                                               ).reindex(ord_ITEM2012).reset_index()
    reord_uni_ua_rich_city = uni_ua_rich_city[uni_ua_rich_city.ITEM2012!="airports"].set_index("ITEM2012"\
                                                                               ).reindex(ord_ITEM2012).reset_index()

    #Empirical univariate, bivariate stats
    df_uni_count_city, df_bi_count_city = generate_stats_emp(d_res_city)
    final_uni_cnt_city = generate_univar_emp(df_uni_count_city)
    boot_final_uni_cnt_city = bootstrapped_generate_univar(df_uni_count_city)
    
    final_bi_cnt_city = generate_bivar_emp(df_bi_count_city)
                    
    test_full_city = get_full_coactivation_emp(final_uni_cnt_city,final_bi_cnt_city)
    #test_nb_city=pd.DataFrame(pd.concat([ua_bi_income_poor_test_city,ua_bi_income_rich_test_city],axis=1)).reset_index()
    #test_nb_city.columns=["pair","poor_nb","rich_nb"]

    # GradCAM Activation Stats Output
    uni_ua_poor_city.to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/{}_poor.csv"\
                                      .format(city,city.lower()),index=False)
    uni_ua_rich_city.to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/{}_rich.csv"\
                                      .format(city,city.lower()),index=False)
    poor_coact_city.to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/poor_coact_{}.csv"\
                                      .format(city,city.lower()),index=False)
    rich_coact_city.to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/rich_coact_{}.csv"\
                                      .format(city,city.lower()),index=False)

    # Emp Stats Output
    test_full_city["poor"].to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/emp_{}_poor_coact.csv"\
                                      .format(city,city.lower()), index=False)
    test_full_city["rich"].to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/emp_{}_rich_coact.csv"\
                                      .format(city,city.lower()), index=False)

    final_uni_cnt_city.to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/{}_cnt.csv"\
                                      .format(city,city.lower()),index=False)
    boot_final_uni_cnt_city.to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/boot_{}_cnt.csv"\
                                      .format(city,city.lower()),index=False)
    #test_nb_city.to_csv(MODEL_OUTPUT_DIR+"2019_income_norm_v2/final_plot_data/{}/emp_bistat.csv".format(city),
    #                     index=False)

if __name__ == '__main__':
    cities = ["Nice","Paris","Lyon","Marseille","Lille"]
    for city in cities:
        print ("Mining {}".format(city))
        extract_valuables_from_city(city);
