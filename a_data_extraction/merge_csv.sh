#!/bin/bash
OutFileName="efs/meta_all.csv"
i=0
for filename in ./*.csv; do
 if [ "$filename"  != "$OutFileName" ] ;
 then
   if [[ $i -eq 0 ]] ; then
      head -1  "$filename" >   "$OutFileName"
   fi
   tail -n +2  "$filename" >>  "$OutFileName"
   i=$(( $i + 1 ))
 fi
done
