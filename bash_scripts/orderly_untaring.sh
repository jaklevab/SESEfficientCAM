set -e
esa_ua_fr_dir="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"

#~/anaconda3/bin/python -c '
#import os
#import pandas as pd
#
#esa_ua_fr_dir = "/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
#data = pd.DataFrame(os.listdir(esa_ua_fr_dir),columns=["tar_file"])
#fr_all = pd.read_csv(esa_ua_fr_dir + "../../sources/FR_UA.csv",sep=";")
#
#for f in os.listdir(esa_ua_fr_dir):
#	if not(f in fr_all["CPP File"].values):
#		comd = "mv " + f +" " + esa_ua_fr_dir + "not_france/"
#		os.system(comd)
#'

cd $esa_ua_fr_dir
for f in $(ls $(echo $esa_ua_fr_dir"*.tar"));do
	echo $f
	tar -xvf $f
	rm -rf $f
done
