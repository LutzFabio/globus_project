#!/bin/bash
#Create file hierarchy for further image magic processing
for i in $(ls -d */*/); do
	if ls ${i}*.webp &>/dev/null; then echo ${i} >> mogrify_filelist.txt; fi 
done

#Create appropriate file hierarchy
cd ../images_small_new
xargs -I {} mkdir -p "{}" < mogrify_filelist.txt
cd ../pictures

#Use imagemagic to resize images and convert them to pngs
#This is just a list of command moglify_command_list.txt
#It can be run by parallel
#cat moglify_command_list.txt | parallel -j 4
while read p; do
  echo "mogrify -resize 500x500\! -quality 100 -path ../images_small_new/$p -format png $p*.webp" >> moglify_command_list.txt
done < mogrify_filelist.txt