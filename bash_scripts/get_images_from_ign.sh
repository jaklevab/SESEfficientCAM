base_dir="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes"
rm -r $base_dir"/data_files/ign/2018/"
image_urls=$(echo $base_dir"/data_files/sources/SPOT_6_Ressources_2018.txt")
filtered_urls=$(echo $base_dir"/data_files/tmp_urls")
cat $image_urls|grep "https" > $filtered_urls

total_lines=$(wc -l $filtered_urls)
download_log="/warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/log_files/log_download.log"

rm $download_log
cd /warehouse/COMPLEXNET/jlevyabi/SATELSES/equirect_proj_test/cnes/data_files/ign/
rm *

while read tile_line; do
        echo $tile_line
        saving_file=$(echo $tile_line|awk -F "/" '{print $(NF-1)}')
        saving_folder=$(echo $saving_file |awk -F "_" '{print $6"/"$5"/"}')
        real_file=$(echo $saving_folder$saving_file|python3 -c 'my_inp=input();print(my_inp.replace("7z-","7z."))')
        mkdir -p $saving_folder
        echo $real_file
        wget $tile_line -O $real_file -a $download_log
done < $filtered_urls

rm $filtered_urls
cd $(echo $base_dir"/bash_scripts")
