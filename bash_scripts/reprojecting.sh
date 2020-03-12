set -e
base_dir="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
batch_file=$(echo $base_dir"batch_data.txt")
log_file=$(echo $base_dir"log_data.txt")
cd $base_dir
rm **/merged_pansharpened_corrected_scaled.TIF

nb_parallel=12
ls -d *_01 | xargs -d '\n' -n $nb_parallel > $batch_file

while read batch_line; do
        for curr_dir in $(echo $batch_line|sed "s/.tgz /.tgz\n/g");do
		scaled_file=$(echo $curr_dir"/merged_pansharpened_corrected.TIF")
		projected_file=$(echo $curr_dir"/merged_pansharpened_corrected_projected.TIF")
                if [ ! -f $scaled_file ]; then
                        echo "File "$scaled_file" not found"
			continue
                fi
		~/anaconda3/envs/bv_env/bin/gdalwarp -t_srs epsg:3035 -r bilinear -co BIGTIFF=YES -co COMPRESS=LZW $scaled_file $projected_file &
                echo "Projected file will be outputted here: "$projected_file
        done
        wait
	for curr_dir in $(echo $batch_line|sed "s/.tgz /.tgz\n/g");do
		scaled_file=$(echo $curr_dir"/merged_pansharpened_corrected.TIF")
		if [ ! -f $pansharpened_file ]; then
			echo "File "$scaled_file" not found"
			continue
		fi
		#rm $scaled_file
	done
done < $batch_file

rm $batch_file
