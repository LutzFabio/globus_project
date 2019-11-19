#!/bin/bash
#Download all data from Globus API
i=200
while [ $i -le 100000000 ]
do
curl -H "Accept: application/json" -H @token_file.txt https://apitest.censhare.globus.ch/products?offset=$i | jq >> data/prod_$((i))_$((i+50)).json
i=$(( $i + 50 ))
done
