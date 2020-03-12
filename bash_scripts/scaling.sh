set -e
base_dir="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
batch_file=$(echo $base_dir"batch_data.txt")
log_file=$(echo $base_dir"log_data.txt")
cd $base_dir
#rm **/merged_pansharpened_corrected.TIF

nb_parallel=21
ls -d *_01 | xargs -d '\n' -n $nb_parallel > $batch_file

while read batch_line; do
        for curr_dir in $(echo $batch_line|sed "s/.tgz /.tgz\n/g");do
                pansharpened_file=$(echo $curr_dir"/merged_pansharpened.TIF")
		scaled_file=$(echo $curr_dir"/merged_pansharpened_corrected.TIF")
                if [ ! -f $pansharpened_file ]; then
                        echo "File "$pansharpened_file" not found"
			continue
                fi
                echo "Scaled file will be outputted here: "$scaled_file
		(~/anaconda3/envs/bv_env/bin/gdalinfo -mm $pansharpened_file > $log_file;
		min=$(cat $log_file | sed -ne 's/.*Computed Min\/Max=//p'| tr -d ' ' | cut -d "," -f 1 | cut -d . -f 1|sort -n|head -1);
		max=$(cat $log_file | sed -ne 's/.*Computed Min\/Max=//p'| tr -d ' ' | cut -d "," -f 2 | cut -d . -f 1|sort -nr|head -1);
		echo "Min: "$min" Max: "$max;
		~/anaconda3/envs/bv_env/bin/gdal_translate $pansharpened_file $scaled_file -b 1 -b 2 -b 3  -scale $min $max 0 65535 -exponent 0.5 -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB -co BIGTIFF=YES) &
        done
        wait
	for curr_dir in $(echo $batch_line|sed "s/.tgz /.tgz\n/g");do
		pansharpened_file=$(echo $curr_dir"/merged_pansharpened.TIF")
		scaled_file=$(echo $curr_dir"/merged_pansharpened_corrected.TIF")
		if [ ! -f $pansharpened_file ]; then
			echo "File "$pansharpened_file" not found"
			continue
		fi
		if [ ! -f $scaled_file ]; then
			echo "File "$scaled_file" not found"
			continue
		fi
		#rm $pansharpened_file
	done
done < $batch_file

rm $batch_file
