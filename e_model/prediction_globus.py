import pandas as pd
import numpy as np
import glob
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, \
    decode_predictions
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, \
  GaussianNoise, Input, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, \
  Adamax, Nadam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing import image


def predict_img(model_path, img_path):
    '''
    Method that aggregates the steps to predict a picture.
    '''

    # Load the latest model.
    model = load_model(model_path)

    # Load the image.
    img = load_img(model.input_shape[1:3], img_path)

    # Predict the image.
    pred_cat, pred_feat = get_pred_df(model.predict(img))

    # Plot.
    plot_img(pred_cat, pred_feat, img)

    return


def plot_img(pred_cat, pred_feat, img):
    '''
    Method to plot the image and the predictions in one.
    '''

    # Create text string.
    txt_str = 'Categories:\n {}: {}\n {}: {}\n {}: {}\n {}: {}\n\n ' \
              'Features:\n {}: {}\n {}: {}\n {}: {}\n {}: {}'.format(
              pred_cat.index[0], pred_cat.iloc[0][0], pred_cat.index[1],
              pred_cat.iloc[1][0], pred_cat.index[2], pred_cat.iloc[2][0],
              pred_cat.index[3], pred_cat.iloc[3][0], pred_feat.index[0],
              pred_feat.iloc[0][0], pred_feat.index[1], pred_feat.iloc[1][0],
              pred_feat.index[2], pred_feat.iloc[2][0], pred_feat.index[3],
              pred_feat.iloc[3][0])

    # Make plot.
    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze())
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.text(0.0, 0.3, txt_str)
    plt.show()

    return


def get_pred_df(pred):
    '''
    Method to get the predictions and the corresponding classes.
    '''

    # Get the latest data frames of the categories and features.
    cat_df = get_latest_csv('categories')
    feat_df = get_latest_csv('features')

    # Combine the categories with the predictions.
    pred_cat = pd.DataFrame(index=cat_df['categories'].values, columns=[
        'prediction'], data=pred[0].squeeze()).sort_values('prediction',
        ascending=False)

    # Combine the features with the predictions.
    pred_feat = pd.DataFrame(index=feat_df['features'].values, columns=[
        'prediction'], data=pred[1].squeeze()).sort_values('prediction',
        ascending=False)

    return pred_cat, pred_feat


def get_latest_csv(kind):
    '''
    Method to get the content of the latest CSV saved. The kind is
    specified by a string.
    '''

    # Get list of the latest category CSV.
    lst_files = glob.glob(model_path + '*' + kind + '*.csv')
    latest_file = max(lst_files, key=os.path.getctime)

    # Read the files.
    df = pd.read_csv(latest_file).drop(['Unnamed: 0'], axis=1)

    return df


def load_img(dims, img_path):
    '''
    Method to load the image specified as array.
    '''

    # Load image and resize.
    img_orig = image.load_img(img_path, target_size=dims)

    # Get the image array.
    img_array = image.img_to_array(img_orig) / 255

    # Expand dimensions.
    img_array_4d = np.expand_dims(img_array, axis=0)

    return img_array_4d


def load_model(model_path):
    '''
    Method that aggregates the steps to load the latest model.
    '''

    # Load the latest model from JSON.
    loaded_model = model_from_json(load_latest_json(model_path))

    # Load the weights.
    loaded_model.load_weights(latest_h5(model_path))

    return loaded_model


def latest_h5(model_path):
    '''
    Method to get the path of the latest HDF5 weight file.
    '''

    # Get latest HDF5 file from output folder.
    lst_files = glob.glob(model_path + '*.h5')
    latest = max(lst_files, key=os.path.getctime)

    return latest


def load_latest_json(model_path):
    '''
    Method to load the JSON model.
    '''

    # Get latest JSON file from output folder.
    lst_files = glob.glob(model_path + '*.json')
    latest = max(lst_files, key=os.path.getctime)

    # Open, assign and close the JSON file.
    j_file = open(latest, 'r')
    j_model = j_file.read()
    j_file.close()

    return j_model


if __name__ == '__main__':

    # Settings.
    model_path = './output_glob/'
    img_path = './images_pred/computer.jpeg'

    # Get prediction.
    predict_img(model_path, img_path)