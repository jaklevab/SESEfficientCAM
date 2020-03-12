set -e
cd /warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/land_ua_esa/FR/
for f in $(ls);do
	echo "Treating file "$f" ..."
	~/anaconda3/envs/bv_env/bin/python /warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/urban_atlas_network_proximal_geometries.py --input $f
	#~/anaconda3/envs/bv_env/bin/python /warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/python_files/urban_atlas_network_touching_geometries.py --input $f
	echo "Done"
done

