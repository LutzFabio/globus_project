import webp
import json
import re
from pandas.io.json import json_normalize

def GetOnlySchuh(x):
    if re.match(r'globus:pim\.category\.(\w+)\.schuhe', x[0]['key']):
        return True
    else:
        return False


with open('prod_1450_1500.json') as json_file:
    data = json.load(json_file)
data=data['items']

f=open("links.txt", "a+") #can save links and run them from bash

df=json_normalize(data, max_level=20)

df['categories']=df['categories'].apply(GetOnlySchuh)
df=df[df['categories']==True]

for index, row in df.iterrows():
    r=row['productItemGroups']
    df_groups = json_normalize(r, max_level=20)
    df_groups=df_groups[~df_groups.media.isna()]
    m=df_groups.media.values[0]
    df_media=json_normalize(m, max_level=20)
    img_links=df_media['formats'].apply(lambda x: x[0]['url'])
    print(img_links)
