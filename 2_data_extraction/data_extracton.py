import webp
import json
import re

with open('prod_1450_1500.json') as json_file:
    data = json.load(json_file)
data=data['items']

f=open("links.txt", "a+")

from pandas.io.json import json_normalize

df=json_normalize(data, max_level=20)

print(df.columns)

'''

for product in data:
    i = 0
    if re.match(r'globus:pim\.category\.(\w+)\.schuhe',product['categories'][0]['key']):
        for groups in product['productItemGroups']:
            v=groups['productItemGroupId']
       # link = product['href']
       # f.write(link + '\n')
       # i+=1
f.close()

'''

print(data)

'''
"key": "globus:pim.category.herren.hosen"
img = webp.load_image('image.webp', 'RGBA')


"categories": [
        {
          "assetId": 518222,
          "key": "globus:pim.category.damenaccessoires.taschen"
        },

'''