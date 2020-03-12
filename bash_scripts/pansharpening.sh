base_dir="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
batch_file=$(echo $base_dir"batch_data.txt")
cd $base_dir
#rm **/merged_pansharpened.TIF
#rm **/merged_pansharpened_corrected.TIF

nb_parallel=20
ls -d *_01 | xargs -d '\n' -n $nb_parallel > $batch_file

while read batch_line; do
        for curr_dir in $(echo $batch_line|sed "s/.tgz /.tgz\n/g");do
                panchromatic_dir=$(ls -d $(echo $curr_dir/"**_PAN"))
                multispectral_dir=$(ls -d $(echo $curr_dir/"**_MUL"))
                panchromatic_file=$(echo $panchromatic_dir"/merged_output.TIF")
                multispectral_file=$(echo $multispectral_dir"/merged_output.TIF")
                pansharpened_file=$(echo $curr_dir"/merged_pansharpened.TIF")
                if [ ! -f $panchromatic_file ]|| [ ! -f $multispectral_file ]; then
                        echo "File "$panchromatic_file"/"$multispectral_file" not found"
			continue
                fi
		if [ ! -f $pansharpened_file ]; then
                        echo "File "$pansharpened_file" already_exists"
                        continue
                fi
		echo "Removing intermediary files"
		#rm $(echo $panchromatic_dir"/*FR110E.TIF")
		#rm $(echo $multispectral_dir"/*FR110E.TIF")
                echo "PS file will be outputted here:"$pansharpened_file
                ~/anaconda3/envs/bv_env/bin/gdal_pansharpen.py $panchromatic_file $multispectral_file $pansharpened_file -r bilinear -co COMPRESS=DEFLATE -co PHOTOMETRIC=RGB  -co BIGTIFF=YES  -b 5 -b 3 -b 2 &
        done
        wait
done < $batch_file
rm $batch_file
