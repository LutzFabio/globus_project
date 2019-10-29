#import webp
import json
import re
from pandas.io.json import json_normalize
import os
import numpy as np
from numpy.core.defchararray import add

def GetOnlySchuh(x):
    if re.match(r'globus:pim\.category\.(\w+)\.schuhe', x[0]['key']):
        return True
    else:
        return False

def extract_data(data):
    names_links=open("names_links.csv", "a+") #can save links and run them from bash

    df=json_normalize(data, max_level=20)

    df['categories']=df['categories'].apply(GetOnlySchuh)
    df=df[df['categories']==True]

    for index, row in df.iterrows():
        r=row['productItemGroups']
        df_groups = json_normalize(r, max_level=20)
        df_groups=df_groups[~df_groups.media.isna()]
        m=df_groups.media.values[0]
        df_media=json_normalize(m, max_level=20)
        img_links=df_media['formats'].apply(lambda x: x[0]['url']).to_frame('links')
        img_links['names']=np.repeat(row['href'].split('/')[-1], len(img_links))
        img_links['names']=img_links['names'] +add(add('_',np.arange(len(img_links)).astype('str')),'.webp')
        img_links.to_csv(names_links, header=False,index=False)
        print(img_links)

def open_data(filename):
    with open(filename, errors='ignore') as json_file:
        data = json.load(json_file)
    data = data['items']
    return data

directory = 'data'
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".json"):
         data=open_data(os.path.join(directory, filename))
         extract_data(data)
         continue
     else:
         continue
