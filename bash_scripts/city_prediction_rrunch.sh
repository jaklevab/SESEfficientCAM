module load cuda/9.0.176_gcc-6.4.0
module load cudnn/7.4_gcc-6.4.0
export GEOS_LIBRARY_PATH="/home/jlevyabi/seacabo/geoanaconda/anaconda3/lib/GEOS_LIBRARY_PATH"
#export SPATIALINDEX_C_LIBRARY=/home/jlevyabi/seacabo/geoanaconda/anaconda3/lib/libspatialindex_c.so

BASE_DIR="/warehouse/COMPLEXNET/jlevyabi/SATELSES/"
CITY_INCOME_DIR="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/outputs/model_data/efficientnet_keras/2019_income_norm/"
cd $BASE_DIR

#tmp_cities=$BASE_DIR"cities_to_do_11000_to_20000.txt"
#cat $CITY_INCOME_DIR"../../../AERIAL_esa_URBAN_ATLAS_FR/city_assoc.csv" |tail -n+2|awk -F "," '{A[$2]+=1}END{for(k in A) print #k";"A[k]}'|sort -t ";" -n -k2,2|awk -F ";" '($2>11000 && $2<20000){$2="";print}'> $tmp_cities

#while read city; 
#do
#    ~/anaconda3/envs/bv_env/bin/python equirect_proj_test/cnes/python_files/keras_experiments/efficientnet/efficientnet_income_prediction_citywise_digitization-ordinal_loss.py -city "$city"
#done < $tmp_cities


tmp_cities=$BASE_DIR"cities_full.txt"
while read city; 
do
    ~/anaconda3/envs/bv_env/bin/python equirect_proj_test/cnes/python_files/keras_experiments/efficientnet/efficientnet_income_prediction_citywise_digitization-ordinal_loss-poorest_richest-stringent.py -city "$city"
done < $tmp_cities



  #~/anaconda3/envs/bv_env/bin/python efficientnet_income_prediction_citywise_digitization.py $city
#efficientnet_income_prediction_citywise_digitization-ordinal_loss.py -city "$city"