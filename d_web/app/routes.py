'''
Main function of the web app, has all routing directions and classification function
'''

#----------------------------Imports
from __future__ import absolute_import, division, print_function, unicode_literals
from flask import Flask
import pandas as pd
import numpy as np
import itertools
import copy
import tensorflow as tf
from flask import render_template, url_for, request, redirect, flash, session
from app import app
from werkzeug.utils import secure_filename
import os
import copy
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image

#--------------------------Some variables
IMAGE_DIMS=(224,224,3) #target dimentions for ResNet50 model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'} #still did not implement it, need to restrict only to upload images
img=[]
#Find all images already in app
for f in os.listdir("app/static/images"):
    if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.webp'):
        img.append(f)

#############################START - ROOT / ##################################
#Forward all requests to page with image name
@app.route('/')
def index():
    '''
    re-roots any request to the page with image name
    '''
    start_img=img[0]
    return redirect('/'+start_img)

#############################IMEGE DISPLAY##################################
#Main page with all stuff
@app.route('/<string:next_img>', methods=['GET', 'POST'])
def update_img(next_img):
    '''
    Main page shows cassification for image
    input: image to show
    output: rendered index.html
    '''

    # Set default model
    used_model='ResNet50_2layers.h5'
    # Check if there was some other model selected
    if request.method == 'POST':
        select = request.form.get('used_model') 
        if select is not None:
            used_model=select
            session['used_model']=select
    if 'used_model' in session:
        used_model=session['used_model'] 

    # Load predictions if they already exists or create new df to keep what was already classified
    if os.path.exists("app/static/cache/model_result.csv"):
        df=pd.read_pickle('app/static/cache/model_result.csv')
    else:
        df=pd.DataFrame(columns=['model','image','final_dict', 'feature_list', 'rest_features','p_color','col_s'])

    # Check if classification already exist in df or calculate a new one    
    if (next_img not in df['image'].values) or (used_model not in df['model'][df['image']==next_img].values):
        final_dict, color, feature_list, rest_features, p_color,col_s=fitCNN('app/static/images/'+next_img,used_model)
        df=df.append({'model':used_model,'image':next_img,'final_dict':final_dict,
            'feature_list': feature_list,'rest_features': rest_features,'p_color': p_color,'col_s':col_s}, ignore_index=True)
    else:
        print('[INFO]: Loading data....')
        print(df)
        df_part=df[(df['image']==next_img)&(df['model']==used_model)]
        final_dict=df_part['final_dict'].values[0]
        feature_list=df_part['feature_list'].values[0]
        rest_features=df_part['rest_features'].values[0]
        p_color=df_part['p_color'].values[0]
        col_s=df_part['col_s'].values[0]

    # Features selection, new image
    if request.method == 'POST':
        # Select features from rest features, not in main feature list
        select = request.form.get('rest_f') 
        if select is not None:
            feature_list.append(select)
            rest_features.remove(select)
        # Upload new image
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

    # Save all
    df.to_pickle('app/static/cache/model_result.csv')

    #Helpers for scroling through images, define next and previous image
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
    # Do not show model in title
    models.remove(used_model)

#----------------------Render web page
    return render_template("index.html",
                            img=url_for('static', filename = img_name),
                            data=feature_list,
                            rest_features=rest_features,
                            next_img=img_next,
                            prev_img=img_prev,
                            models=models,
                            used_model=used_model,
                            color=p_color,
                            col_s=col_s,
                            final_dict=final_dict)

################################SANBOX#######################
#I dont want to break what works already. To test pages:
@app.route('/test', methods=['GET', 'POST'])
def test():
    '''
    Does whatever I want to test
    input: nothing
    output: nothing
    '''

    cc=['50% - Hemd', '30% - Kleid','1% - Jacke']
    cat_2_sel=cc[0]
    if request.method == 'POST':
        print('#########################')
        select = request.form.get('rating') 
        cat_2_sel=select  
        print(select) 
    cc.remove(cat_2_sel)
        

    return render_template("test.html",cat_2=cc,cat_2_sel=cat_2_sel)

#################################CNN CLASSIFICATION####################
#Helper to calculate probabilities from the model
def fitCNN(image_path,modelname):
    '''
    Function fits CNN model to predict image category and features
    input: path to image, should be smth like statec/images/name.png
    output: predictions
    '''
    #from app import best_model as model
    print('[INFO]: Loading data....')
    #read supporting csvs
    features=pd.read_csv('app/static/models/features_'+modelname[:-3]+'.csv')
    categories=pd.read_csv('app/static/models/categories_'+modelname[:-3]+'.csv')
    colors=pd.read_csv('app/static/models/colors.csv')
    feature_dict=pd.read_csv('app/static/models/rename_features.csv',sep=';')

    #Rename some encoded things
    categories_dict={ 'blusentuniken':'blusen, tuniken',
                     'pulloverstrick':'pullover strick',
                     'topshirtssweats':'top shirts, sweats',
                     'shirtstopssweats':'top shirts, sweats',
                     'strickpullover':'pullover strick',
                     '34armshirt':'3/4 armshirt',
                     }
    categories['categories']=categories['categories'].replace(categories_dict, regex=True)
    feature_dict=pd.Series(feature_dict['decoding'].values,index=feature_dict['features']).to_dict()
    features['features']=features['features'].map(feature_dict)
    do_not_keep=['l1','l16','l18','l2']
    
    #Extract important things - for categories
    hirarchy=pd.DataFrame(categories['categories'].str.split('_').tolist(),
                                   columns = ['cat_1','cat_2','cat_3'])

    ##Extract important things - for features
    colors=colors['colors'].values.tolist()
    pos_color=features['features'].str.contains('|'.join(colors))
    features=features['features'].values

    print('[INFO]: Fitting model data....')
    model = tf.keras.models.load_model('app/static/models/'+modelname)

    # Load image and resize.
    img_orig = load_img(image_path, target_size=(224,224))
    # Get the image array.
    # img_array = image.img_to_array(img_orig) / 255
    img_array = img_to_array(img_orig)
    # Expand dimensions.
    img_array_4d = np.expand_dims(img_array, axis=0)
    # Preprocess the image.
    image = preprocess_input(img_array_4d)


    pred = model.predict([image])

    print('[INFO]: Fixing outputs....')
    #-------------------work with categories predictions
    cat_list=pred[0][0]
    if len(cat_list)>17:
        hirarchy['prob']=cat_list
    else:
        hirarchy=hirarchy.groupby(['cat_2','cat_1']).agg({'cat_1':'first','cat_2':'first'}).reset_index(drop=True)
        hirarchy['prob']=cat_list
    #First hierarchy
    cat_1_prob=hirarchy.groupby('cat_1').agg({'cat_1':'first','prob':'sum'}).reset_index(drop=True)
    cat_1_prob['prob']=cat_1_prob['prob']*100
    cat_1_prob=cat_1_prob.sort_values(by='prob',ascending=False)
    cat_1_prob['cat1_name']=cat_1_prob[['cat_1','prob']].apply(lambda x: '%0.0f%% - %s'%(x['prob'],x['cat_1']),axis=1)
    #Second hierarchy
    cat_2_prob=hirarchy.groupby(['cat_1','cat_2']).agg({'cat_1':'first','cat_2':'first','prob':'sum'}).reset_index(drop=True)
    cat_2_prob['prob']=cat_2_prob['prob']*100
    cat_2_prob=cat_2_prob.sort_values(by='prob',ascending=False)
    cat_2_prob=cat_2_prob.iloc[:4,:]
    cat_2_prob['cat2_name']=cat_2_prob[['cat_2','prob']].apply(lambda x: '%0.0f%% - %s'%(x['prob'],x['cat_2']),axis=1)
    #Third hierarchy
    if len(cat_list)>17:
        cat_3_prob=hirarchy.merge(cat_2_prob,on=['cat_1','cat_2'])
        cat_3_prob=cat_3_prob.sort_values(by='prob_x',ascending=False)
        cat_3_prob=cat_3_prob.iloc[:5,:]
        cat_3_prob=cat_3_prob.drop(columns=['cat2_name','prob_y'])
        cat_3_prob['prob_x']=cat_3_prob['prob_x']*100
        cat_3_prob['cat3_name']=cat_3_prob[['cat_3','prob_x']].apply(lambda x: '%0.1f%% - %s'%(x['prob_x'],x['cat_3']),axis=1)

        cat_2_prob=cat_2_prob[cat_2_prob['cat_2'].str.contains('|'.join(cat_3_prob['cat_2'].values.tolist())) & 
                    cat_2_prob['cat_1'].str.contains('|'.join(cat_3_prob['cat_1'].values.tolist()))]
        cat_1_prob=cat_1_prob[cat_1_prob['cat_1'].str.contains('|'.join(cat_2_prob['cat_1'].values.tolist()))]
    else:
        cat_3_prob=None
    # Combine into big hierarchy dictionary
    final_dict={}
    for h1 in range(len(cat_1_prob)):
        col='white' if h1>0 else '#e6e6ff'
        final_dict[cat_1_prob['cat_1'].iloc[h1]]={'name':cat_1_prob['cat1_name'].iloc[h1], 'color':col, 'subh':[]}
        cat_2_prob_h2=cat_2_prob[cat_2_prob['cat_1']==cat_1_prob['cat_1'].iloc[h1]]
        subh2=[]
        for h2 in range(len(cat_2_prob_h2)):
            col2 = '#e6e6ff' if h2 == 0 and h1==0 else 'white'
            subh2.append({'name':cat_2_prob_h2['cat2_name'].iloc[h2], 'color':col2, 'subh':[]})
            if cat_3_prob is not None:
                cat_3_prob_h3 = cat_3_prob[(cat_3_prob['cat_1'] == cat_1_prob['cat_1'].iloc[h1])&(cat_3_prob['cat_2'] == cat_2_prob['cat_2'].iloc[h2])]
                subh3=[]
                for h3 in range(len(cat_3_prob_h3)):
                    col3 = '#e6e6ff' if h2 == 0 and h1==0 and h3==0 else 'white'
                    subh3.append({'name': cat_3_prob_h3['cat3_name'].iloc[h3], 'color': col3})
                subh2[h2]['subh']=subh3

        final_dict[cat_1_prob['cat_1'].iloc[h1]]['subh']=subh2

    #----------------------work with features predictions
    feat_list=pred[1][0]
    #color
    p_color=feat_list[pos_color]
    p_col=features[pos_color]
    pos=p_color.argsort()[-1:][::-1]
    color='%0.0f%% - %s'%(np.max(p_color)*100,p_col[pos].tolist()[0])
    # other features
    feature_list=['%0.0f%% - %s'%(x*100,y) for x,y in zip(feat_list[feat_list>=0.7],features[feat_list>=0.7])]
    rest_features=['%0.0f%% - %s'%(x*100,y) for x,y in zip(feat_list[feat_list<0.7],features[feat_list<0.7])]

    for f in do_not_keep:
        try: 
            feature_list.remove(f)
        except: print('not there')
    for f in do_not_keep:
        try: 
            rest_features.remove(f)
        except: print('not there')

    # Color from segmentation
    img = Image.open(image_path)

    mask=np.array(img.split()[-1])

    if len(mask[mask==0])>200:
        mask[mask > 0] = 1
    else:
        mask[mask<=100]=1
        mask[mask>100]=0


    r=np.array(img.split()[0])*mask
    g=np.array(img.split()[1])*mask
    b=np.array(img.split()[2])*mask

    r=round(np.median(r[r>0])).astype(int)
    g=round(np.median(g[g>0])).astype(int)
    b=round(np.median(b[b>0])).astype(int)

    col_s='#%02x%02x%02x' % (r, g, b)

    return final_dict, color, feature_list, rest_features, color,col_s


