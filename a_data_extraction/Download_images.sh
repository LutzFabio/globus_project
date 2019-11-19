#!/bin/bash
#Create list for further parallel run
#Download images based on data from jsons in parallel
#cat run_list.txt | parallel -j 4

for i in $(ls *.json); do
echo "python json_parsing.py ${i}" >> run_list.txt
done


