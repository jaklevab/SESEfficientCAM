base_dir="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/esa/URBAN_ATLAS/"
batch_file=$(echo $base_dir"sec_batch_data.txt")
cd $base_dir

#set -e

nb_parallel=18
ls -d *_01 | xargs -d '\n' -n $nb_parallel > $batch_file

while read batch_line; do
        for curr_dir in $(echo $batch_line|sed "s/.tgz /.tgz\n/g");do
                panchromatic_dir=$(ls -d $(echo $curr_dir/"**_PAN"))
                multispectral_dir=$(ls -d $(echo $curr_dir/"**_MUL"))
                #panchromatic_file=$(echo $panchromatic_dir"/merged_output.TIF")
                #multispectral_file=$(echo $multispectral_dir"/merged_output.TIF")
                #pansharpened_file=$(echo $curr_dir"/merged_pansharpened.TIF")
		#pansharpened_corr_file=$(echo $curr_dir"/merged_pansharpened_corrected.TIF")
		projected_file=$(echo $curr_dir"/merged_pansharpened_corrected_projected.TIF")
                #if [ ! -f $panchromatic_file ]|| [ ! -f $multispectral_file ]; then
                #        echo "File "$panchromatic_file"/"$multispectral_file" not found"
		#	continue
                #fi
		#if [ ! -f $pansharpened_file ]; then
                #        echo "File "$pansharpened_file" not found"
                #        continue
                #fi
		if [ ! -f $projected_file ]; then
                        echo "File "$projected_file" not found"
                        continue
                fi
		echo "Removing intermediary files"
		#ls $(echo $panchromatic_dir/"*P001.TIF")
		rm $(echo $panchromatic_dir/"*P001.TIF")
		rm $(echo $multispectral_dir/"*P001.TIF")
		#rm $multispectral_file
		#rm $pansharpened_file
        done
        wait
done < $batch_file

rm $batch_file
