module load cuda/9.0.176_gcc-6.4.0
module load cudnn/7.4_gcc-6.4.0
export GEOS_LIBRARY_PATH="/home/jlevyabi/seacabo/geoanaconda/anaconda3/lib/GEOS_LIBRARY_PATH"

cd /warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/keras_experiments/efficientnet/

~/anaconda3/envs/bv_env/bin/efficientnet_income_prediction_citywise_digitization-ordinal_loss-hyperparametrizing.py \
-city Rennes -lr 1e-2 -epochs 10 -spe 200 -lr_pat 5 -cv 5

~/anaconda3/envs/bv_env/bin/efficientnet_income_prediction_citywise_digitization-ordinal_loss-hyperparametrizing.py \
-city Rennes -lr 1e-3 -epochs 15 -spe 200 -lr_pat 5 -cv 5

~/anaconda3/envs/bv_env/bin/efficientnet_income_prediction_citywise_digitization-ordinal_loss-hyperparametrizing.py \
-city Rennes -lr 1e-4 -epochs 20 -spe 200 -lr_pat 5 -cv 5

~/anaconda3/envs/bv_env/bin/efficientnet_income_prediction_citywise_digitization-ordinal_loss-hyperparametrizing.py \
-city Rennes -lr 1e-5 -epochs 30 -spe 200 -lr_pat 5 -cv 5

~/anaconda3/envs/bv_env/bin/efficientnet_income_prediction_citywise_digitization-ordinal_loss-hyperparametrizing.py \
-city Rennes -lr 1e-4 -epochs 30 -spe 100 -lr_pat 5 -cv 5

~/anaconda3/envs/bv_env/bin/efficientnet_income_prediction_citywise_digitization-ordinal_loss-hyperparametrizing.py \
-city Rennes -lr 1e-4 -epochs 30 -spe 500 -lr_pat 5 -cv 5
