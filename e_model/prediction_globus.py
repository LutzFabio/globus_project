import pandas as pd
import numpy as np
import glob
import os
import cv2
from PIL import Image

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


def predict_img(model_path, img_path):
    '''
    Method that aggregates the steps to predict a picture.
    '''

    # Load the latest model.
    model = load_model(model_path)

    # Load the image.
    img = load_img(model.input_shape[1:3], img_path)

    a = 1


    return


def load_img(dims, img_path):
    '''
    Method to load the image specified as array.
    '''

    # Read the image.
    img_orig = Image.open(img_path)

    # Resize the image.
    img_resized = img_orig.resize(dims)

    # Resize the image.
    img_rsz = cv2.resize(img_orig, dims)

    return


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
    img_path = './images_pred/test_1.jpg'

    # Get prediction.
    predict_img(model_path, img_path)