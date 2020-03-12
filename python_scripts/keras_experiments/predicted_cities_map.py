from geopandas import GeoDataFrame
import geopandas as gpd
import pandas as pd 
import numpy as np
import os
from matplotlib.cm import inferno,RdBu_r
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/warehouse/COMPLEXNET/jlevyabi/"
SAT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
CENSUS_DIR = BASE_DIR + 'REPLICATE_LINGSES/data_files/census_data/'
UA_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/"
OUTPUT_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/data_files/outputs/AERIAL_esa_URBAN_ATLAS_FR/"
new_RES_DIR = BASE_DIR + "SATELSES/equirect_proj_test/cnes/"\
                        +"data_files/outputs/model_data/efficientnet_keras/2019_income_norm_v2/"
NB_SES_VALUES = 5
inter_col = np.linspace(0,1,NB_SES_VALUES)
dic_color = {str(k):RdBu_r(inter_col[k]) for k in range(NB_SES_VALUES)}

sns.set_style("white")
plt.style.use('bmh')

print("Generating Income Classes")
idINSPIRE2GEOM = GeoDataFrame.from_file(BASE_DIR + 'INSEE/2019/200m/shps/Filosofi2015_carreaux_200m_metropole.shp')
idINSPIRE2GEOM["income"] = idINSPIRE2GEOM["Ind_snv"]/idINSPIRE2GEOM["Ind"]
idINSPIRE2GEOM.rename({"IdINSPIRE":"idINSPIRE"},axis=1,inplace=True)
city_assoc = pd.read_csv(OUTPUT_DIR + "city_assoc.csv")
full_im_df_ua = gpd.GeoDataFrame(pd.merge(city_assoc,idINSPIRE2GEOM,on="idINSPIRE"),crs=idINSPIRE2GEOM.crs);

val_min = lambda x : np.percentile(x,0)
val_per20 = lambda x : np.percentile(x,20)
val_per40 = lambda x : np.percentile(x,40)
val_per60 = lambda x : np.percentile(x,60)
val_per80 = lambda x : np.percentile(x,80)
val_max = lambda x : np.percentile(x,100)

val_min.__name__ = 'qmin'
val_per20.__name__ = 'q20'
val_per40.__name__ = 'q40'
val_per60.__name__ = 'q60'
val_per80.__name__ = 'q80'
val_max.__name__ = 'qmax'

ses_city_intervals = full_im_df_ua.groupby("FUA_NAME")[["income"]].agg(
    [val_min,val_per20,val_per40,val_per60,val_per80,val_max]
)
df_cities = []
for city in list(ses_city_intervals.index):
    city_df_new = full_im_df_ua[full_im_df_ua.FUA_NAME==city]
    city_df_new.dropna(subset=["income"],inplace=True)
    income = city_df_new.income
    class_thresholds = ses_city_intervals.loc[city]["income"].values
    x_to_class = np.digitize(income,class_thresholds)
    x_to_class[x_to_class==np.max(x_to_class)] = NB_SES_VALUES
    city_df_new["treated_citywise_income"] = [ str(y-1) for y in x_to_class ] 
    df_cities.append(city_df_new)

full_im_df_ua = gpd.GeoDataFrame(pd.concat(df_cities,axis=0),crs=full_im_df_ua.crs).sort_index()

print("Getting Predicted Classes")
predicted_cities = [city  for city in os.listdir(new_RES_DIR) if "preds" in os.listdir(new_RES_DIR+city)
                    if "full_whole_predictions.csv" in os.listdir(new_RES_DIR+city+"/preds")]
predicted_values = pd.concat([pd.read_csv(new_RES_DIR+city+"/preds/full_whole_predictions.csv") 
                     for city in predicted_cities],axis=0).sort_index()
predicted_values["pred_class"] = predicted_values["pred_class"].astype(str)
full_mat = gpd.GeoDataFrame(pd.merge(full_im_df_ua[["idINSPIRE","FUA_NAME","treated_citywise_income","geometry"]],
                                     predicted_values,on="idINSPIRE"),crs=full_im_df_ua.crs)

print("Mapping")
with plt.style.context('dark_background'):
    for city in predicted_cities:
        print("Treating {}".format(city))
        f,axes = plt.subplots(1,2,figsize=(12,8))
        city_df = full_mat[full_mat.FUA_NAME == city].to_crs({"init":"epsg:3035"})
        for ses_class,ses_color in dic_color.items():
            city_df[city_df.treated_citywise_income==ses_class].plot(
                "treated_citywise_income",color=ses_color,edgecolor=None,linewidth=0.0,ax=axes[0]);
        axes[0].axis('off');
        axes[0].set_title('Original',fontsize=18);
        for ses_class,ses_color in dic_color.items():
            city_df[city_df.pred_class==ses_class].plot(
                "pred_class",color=ses_color,edgecolor=None,linewidth=0.0,ax=axes[1]);
        axes[1].axis('off');
        axes[1].set_title('Predicted',fontsize=18);
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=0, vmax=NB_SES_VALUES-1))
        sm._A = []
        cbaxes = f.add_axes([0.42, .2, 0.16, 0.01]) 
        cbar=plt.colorbar(sm,extend='both',ticks=range(NB_SES_VALUES),cax = cbaxes,orientation="horizontal",
                      fraction=.01,);
        f.suptitle(city,fontsize=25,x=0.82,y=0.15,style='italic',);
        cbar.ax.tick_params(labelsize=20)
        f.savefig("/warehouse/COMPLEXNET/jlevyabi/tmp/"+city+"_income_pred_ord_loss.pdf",bbox_inches='tight', pad_inches=0.1);
