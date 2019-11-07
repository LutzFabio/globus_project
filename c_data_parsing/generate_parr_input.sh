#!/bin/bash

path_var=$1/*.json

for entry in $path_var
do
  echo "python json_parsing.py ${entry:2}" >> run_list.txt
done
