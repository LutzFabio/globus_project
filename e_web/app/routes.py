
from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import copy
import tensorflow as tf
from flask import render_template, url_for, request, redirect, flash
from app import app
from werkzeug.utils import secure_filename
import os
import PIL
import copy
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

IMAGE_DIMS=(224,224,3)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
img=[]

for f in os.listdir("app/static/images"):
    if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.webp'):
        img.append(f)

#Forwar all requests to page with image name
@app.route('/')
def index():
    start_img=img[0]
    return redirect('/'+start_img)

#Main page with all stuff
@app.route('/<string:next_img>', methods=['GET', 'POST'])
def update_img(next_img):

    #Classify/load predictions

    if os.path.exists("app/static/cache/model_result.csv"):
        df=pd.read_pickle('app/static/cache/model_result.csv')
    else:
        df=pd.DataFrame(columns=['image','prob_c1', 'prob_c2', 'p_color', 'feature_list', 'rest_features','dd','col_s'])

    if next_img not in df['image'].values:
        prob_c1, prob_c2, p_color, feature_list, rest_features, dd, col_s=fitCNN('app/static/images/'+next_img)
        df=df.append({'image':next_img,'prob_c1':prob_c1,
            'prob_c2': prob_c2,'p_color': p_color,'feature_list': feature_list,'rest_features': rest_features,
            'dd':dd, 'col_s':col_s}, ignore_index=True)
    else:
        print('[INFO]: Loading data....')
        prob_c1=df['prob_c1'][df['image']==next_img].values[0]
        prob_c2=df['prob_c2'][df['image']==next_img].values[0]
        feature_list=df['feature_list'][df['image']==next_img].values[0]
        rest_features=df['rest_features'][df['image']==next_img].values[0]
        p_color=df['p_color'][df['image']==next_img].values[0]
        dd=df['dd'][df['image']==next_img].values[0]
        col_s=df['col_s'][df['image']==next_img].values[0]

    cat_2_sel=prob_c2[0]
    cat_1_sel=prob_c1[0]


    df.to_pickle('app/static/cache/model_result.csv')

    if request.method == 'POST':
        select = request.form.get('cat_2') 
        if select is not None:
            cat_2_sel=select
        select = request.form.get('cat_1') 
        if select is not None:
            cat_1_sel=select   

        if 'file' not in request.files:
            flash('No file part')
        else:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
            else:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                img.insert(0, filename)
                return redirect(url_for('index'))

    tree_up=copy.deepcopy(prob_c1)
    prob_c2.remove(cat_2_sel)
    prob_c1.remove(cat_1_sel)

    #Helpers for scroling through images

    img_name='images/'+next_img

    idx=img.index(next_img)+1
    idx= idx if idx<len(img) else 0
    img_next=img[idx]

    idx=img.index(next_img)-1
    idx= idx if idx>=0 else len(img)-1
    img_prev=img[idx]

    #List available models
    models=[]
    for file in os.listdir("app/static/models"):
        if file.endswith(".h5"):
            models.append(file)

    used_model=models[0]

#----------------------Return all
    return render_template("index.html",
                            img=url_for('static', filename = img_name),
                            data=feature_list,
                            rest_features=rest_features,
                            next_img=img_next,
                            prev_img=img_prev,
                            cat_2=prob_c2,
                            cat_2_sel=cat_2_sel,
                            cat_1=prob_c1,
                            cat_1_sel=cat_1_sel,
                            models=models,
                            used_model=used_model,
                            color=p_color,
                            tree_up=tree_up,
                            tree_dict=dd,
                            col_s=col_s)


#To test pages
@app.route('/test', methods=['GET', 'POST'])
def test():
    cc=['50% - Hemd', '30% - Kleid','1% - Jacke']
    cat_2_sel=cc[0]
    if request.method == 'POST':
        print('#########################')
        select = request.form.get('rating') 
        cat_2_sel=select  
        print(select) 
    cc.remove(cat_2_sel)
        

    return render_template("test.html",cat_2=cc,cat_2_sel=cat_2_sel)

#Helper to calculate probabilities from the model
def fitCNN(image_path):
    print('[INFO]: Loading data....')
    #read supporting csvs
    features=pd.read_csv('app/static/models/features_20191114-164753.csv')
    categories=pd.read_csv('app/static/models/categories_20191114-112913.csv')
    colors=pd.read_csv('app/static/models/colors.csv')

    #Extract important things - for categories
    categories['categories']=categories['categories'].apply(lambda x: ','.join(x.split('_')[:2]))
    categories=categories.groupby('categories').first().reset_index()
    pos_damen=categories['categories'].str.contains('damen')
    pos_herren=categories['categories'].str.contains('herren')
    pos_kinder=categories['categories'].str.contains('kinder')
    pos_damenaccessoires=categories['categories'].str.contains('damenaccessoires')
    categories=categories['categories'].values

    ##Extract important things - for features
    colors=colors['colors'].values.tolist()
    pos_color=features['features'].str.contains('|'.join(colors))
    features=features['features'].values

    print('[INFO]: Fitting model data....')
    model = tf.keras.models.load_model('app/static/models/ResNet50_2layers.h5')

    image = cv2.imread(image_path)
    image = cv2.resize(image,IMAGE_DIMS[:2])
    image = image.reshape(1, 224, 224, 3)
    image = preprocess_input(image)

    pred = model.predict([image])

    print('[INFO]: Fixing outputs....')
    #work with categories predictions
    cat_list=pred[0][0]
    #First hierarchy
    prob_damen=np.sum(cat_list[pos_damen] ).item()*100
    prob_herren=np.sum(cat_list[pos_herren] ).item()*100
    prob_kinder=np.sum(cat_list[pos_kinder] ).item()*100
    prob_acc=np.sum(cat_list[pos_damenaccessoires] ).item()*100
    prob_c1={'Damen':prob_damen,'Herren':prob_herren,'Kinder':prob_kinder,'Damen accessoires':prob_acc}
    prob_c1=sorted(prob_c1.items(), key=lambda x: x[1], reverse=True)
    prob_c1=['%0.0f%% - %s'%(x[1],x[0]) for x in prob_c1]
    #Second hierarchy
    dd={'damen':[],'herren':[],'kinder':[],'damenaccessoires':[]}
    if np.all(cat_list < 0.9):
        pos=cat_list.argsort()[-3:][::-1]
        prob_c2=['%0.1f%% %s'%(cat_list[p]*100,categories[p].split(',')[1]) for p in pos]
        for p in pos:
            dd[categories[p].split(',')[0]].append('%0.1f%% %s' % (cat_list[p] * 100, categories[p].split(',')[1]))
    else:
        prob=cat_list[cat_list>=0.9]
        cc=categories[cat_list >= 0.9].tolist()[0].split(',')
        prob_c2='%0.0f%% - %s'%(prob[0]*100,cc[1])


    dd['Damen'] = dd.pop('damen')
    dd['Herren'] = dd.pop('herren')
    dd['Kinder'] = dd.pop('kinder')
    dd['Damen accessoires'] = dd.pop('damenaccessoires')

    print(dd)
    #work with features predictions
    feat_list=pred[1][0]
    #color
    p_color=feat_list[pos_color]
    p_col=features[pos_color]
    pos=p_color.argsort()[-1:][::-1]
    color='%0.0f%% - %s'%(np.max(p_color)*100,p_col[pos].tolist()[0])
    #other features
    feature_list=['%0.0f%% - %s'%(x*100,y) for x,y in zip(feat_list[feat_list>=0.7],features[feat_list>=0.7])]
    rest_features=['%0.0f%% - %s'%(x*100,y) for x,y in zip(feat_list[feat_list<0.7],features[feat_list<0.7])]

    #Color from segmentation
    img = Image.open(image_path)

    mask=np.array(img.split()[-1])

    mask[mask<=100]=1
    mask[mask>100]=0


    r=np.array(img.split()[0])*mask
    g=np.array(img.split()[1])*mask
    b=np.array(img.split()[2])*mask

    r=round(np.median(r[r>0])).astype(int)
    g=round(np.median(g[g>0])).astype(int)
    b=round(np.median(b[b>0])).astype(int)

    col_s='#%02x%02x%02x' % (r, g, b)

    return prob_c1, prob_c2, color, feature_list, rest_features, dd, col_s


