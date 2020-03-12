module load cuda/9.0.176_gcc-6.4.0
module load cudnn/7.4_gcc-6.4.0
export GEOS_LIBRARY_PATH="/home/jlevyabi/seacabo/geoanaconda/anaconda3/lib/GEOS_LIBRARY_PATH"

cd /warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/keras_experiments/efficientnet/

 ~/anaconda3/envs/bv_env/bin/python gradcaming_urban_areas_citywise_mem_friendly_optimized.py -city Nice -model_dir 2019_income_norm_v2 -gpu_id 0
