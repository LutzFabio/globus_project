import pandas as pd
import numpy as np
import os
import glob
import time
import math
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import compress
from ast import literal_eval


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


class GlobusCNN:
    '''
    Build a class object that handles all the tasks the CNN
    should be performing.
    '''

    # Settings.
    im_resc =             1.0 / 255
    rot_range =           180
    width_shift =         0.1
    height_shift =        0.1
    shear_rng =           0.1
    zoom_rng =            [0.9, 1.5]
    hor_flip =            True
    vert_flip =           True
    fill_mod =            'nearest'
    batch_size_train =    10
    batch_size_test =     3
    meta_file =           '/home/fabiolutz/propulsion/globus_project/' \
                          'd_data_cleaning/meta_clean_red_train.csv'
    path_images =         '/home/fabiolutz/propulsion/globus_project/' \
                          'e_model/images_small_new/'
    # meta_file =           '/home/ubuntu/efs/meta_clean_red_train.csv'
    # path_images =         '/home/ubuntu/efs/images_small_new/'
    sample_size =         200
    test_size =           0.2
    rand_state=           200
    model_str =           'ResNet50'
    model_input=          (224, 224, 3)
    layer_old =           'conv5_block3_out'
    layer_old_train =     None # 'conv5_block3'
    activation_dense =    'relu'
    size_dense =          1024
    size_output =         1
    dropout =             0.2
    activation_last =     'sigmoid'
    optimizer_str =       'Adam'
    learning_rate =       0.01
    loss =                'categorical_crossentropy'
    metrics =             ['categorical_accuracy']
    epochs =              2
    epoch_steps =         2


    def __init__(self, load=False):
        '''
        Initiate the class instance.
        '''

        # Load meta file.
        self.meta_data = pd.read_csv(self.meta_file)

        # Initiate the optimizer.
        if self.optimizer_str == 'Adam':
            self.optimizer = Adam(lr=self.learning_rate)

        elif self.optimizer_str == 'Adagrad':
            self.optimizer = Adagrad(lr=self.learning_rate)

        elif self.optimizer_str == 'Adadelta':
            self.optimizer = Adadelta(lr=self.learning_rate)

        else:
            raise ValueError('Optimizer not implemented yet!')

        # Define whether to train or to load a model and evaluate it.
        if not load:
            # Define the model to train.
            if self.model_str == 'ResNet50':
                self.model = ResNet50(weights='imagenet', include_top=False,
                                      input_shape=self.model_input)

            else:
                raise ValueError('Model not implemented yet!')

            # Train it.
            self.train()

            # Evaluate.
            self.evaluate_train()

        else:
            # Load the JSON model and the weights.
            loaded_model = model_from_json(self.load_json_model())
            loaded_model.load_weights(self.latest_h5())
            self.model = loaded_model

            # Augment the images.
            self.augment_images()

            # Compile the model.
            loaded_model.compile(optimizer=self.optimizer,
                                 loss=self.loss,
                                 metrics=self.metrics)
            self.model_new = loaded_model

            # Evaluate.
            self.evaluate_load()


    def train(self):
        '''
        Method that trains a new model and takes care of the
        tasks needed in this regard.
        '''

        # Augment the pictures.
        self.train_gen_cat, self.test_gen_cat, self.train_gen_feat, \
        self.test_gen_feat = self.augment_images()

        # Create new model.
        self.model_cat = self.create_new_model()

        # Compile the model.
        self.model_cat.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)

        # Train the model.
        self.model_trained = self.model_cat.fit_generator(
            generator=self.train_gen, epochs=self.epochs,
            steps_per_epoch=self.epoch_steps, validation_data=self.test_gen,
            class_weight=self.class_wgt(self.train_gen.classes),
            validation_steps=self.steps_test())

        return


    def evaluate_train(self):
        '''
        Method to evaluate the trained model.
        '''

        # Plot and save an image of the accuracy to the output folder.
        self.plot_result(self.model_trained)

        # Save the accuracy in a text file.
        self.save_accuracy(self.model_new)

        # Save model.
        self.save_model(self.model_new)

        return


    def evaluate_load(self):
        '''
        Method to evaluate the loaded model.
        '''

        # Save the accuracy in a text file.
        self.save_accuracy(self.model_new)

        return


    def latest_h5(self):
        '''
        Method to the the path of the latest HDF5 weight file.
        '''

        # Get latest HDF5 file from output folder.
        lst_files = glob.glob('./output_hier/*.h5')
        latest = max(lst_files, key=os.path.getctime)

        return latest


    def load_json_model(self):
        '''
        Method to load the JSON model.
        '''

        # Get latest JSON file from output folder.
        lst_files = glob.glob('./output_hier/*.json')
        latest = max(lst_files, key=os.path.getctime)

        # Open, assign and close the JSON file.
        j_file = open(latest, 'r')
        j_model = j_file.read()
        j_file.close()

        return j_model


    def steps_test(self):
        '''
        Method to calculate the test steps.
        '''

        return self.test_gen.n / self.batch_size_test


    def class_wgt(self, classes):
        '''
        Method to calculate the class weights, such that imbalanced classes
        could be handled.
        '''

        wgt = compute_class_weight(class_weight='balanced',
                                   classes=np.unique(classes),
                                   y=classes)

        return wgt


    def create_new_model(self):
        '''
        Method that creates a new model based on an existing one
        (transfer learning)
        '''

        # Set all the layers in the original model to non-trainable.
        self.model.trainable = False

        # Check if something should be re-trained in the original
        # model.
        if self.layer_old_train is not None:
            for layer in self.model.layers:
                trainable = (self.layer_old_train in layer.name)
                layer.trainable = trainable

        # Create the categorical model.
        flat_cat = Flatten()(self.model.outputs[0])
        class_cat = Dense(self.size_dense,
                          activation=self.activation_dense)(flat_cat)
        output_cat = Dense(len(np.unique(self.train_gen_cat.classes)),
                           activation=self.activation_dense)(class_cat)
        activ_cat = Activation(self.activation_last)(output_cat)

        # Create the feature model.
        flat_feat = Flatten()(self.model.outputs[0])
        class_feat = Dense(self.size_dense,
                          activation=self.activation_dense)(flat_feat)
        output_feat = Dense(len(np.unique(self.train_gen_feat.classes)),
                           activation=self.activation_dense)(class_feat)
        activ_feat = Activation(self.activation_last)(output_feat)

        # Combine the model.
        model = Model(inputs=self.model.inputs,
                      outputs=[activ_cat, activ_feat])

        return model


    def augment_images(self):
        '''
        Method that takes care of the image augmentation.
        '''

        # Get image dimensions.
        try:
            im_dims = self.model.layers[0].output_shape[0][1:3]
        except:
            im_dims = self.model.layers[0].input_shape[1:3]

        # Get images for train and test.
        self.train_df_cat, self.test_df_cat, self.train_df_feat, \
        self.test_df_feat  = self.get_image_dfs()

        # Initiate the train ImageDataGenerator.
        gen_train_data = ImageDataGenerator(
            rescale=self.im_resc,
            rotation_range=self.rot_range)
            # width_shift_range=self.width_shift,
            # height_shift_range=self.height_shift,
            # shear_range=self.shear_rng,
            # zoom_range=self.zoom_rng,
            # horizontal_flip=self.hor_flip,
            # vertical_flip=self.vert_flip,
            # fill_mode=self.fill_mod)

        # Inititate the train generators.
        gen_train_cat = gen_train_data.flow_from_dataframe(
            self.train_df_cat, batch_size=self.batch_size_train,
            x_col='img_path', y_col='img_class', target_size=im_dims,
            shuffle=True)
            #save_to_dir=self.path_images + 'augmented/',
            #save_format='png')

        gen_train_feat = gen_train_data.flow_from_dataframe(
            self.train_df_feat, batch_size=self.batch_size_train,
            x_col='img_path', y_col='features_clean', target_size=im_dims,
            shuffle=True)
            #save_to_dir=self.path_images + 'augmented/',
            #save_format='png')

        # Initiate the test ImageDataGenerator.
        gen_test_data = ImageDataGenerator(
            rescale=self.im_resc)

        # Inititate the train generators.
        gen_test_cat = gen_test_data.flow_from_dataframe(
            self.test_df_cat, batch_size=self.batch_size_test,
            x_col='img_path', y_col='img_class', target_size=im_dims,
            shuffle=False)

        gen_test_feat = gen_test_data.flow_from_dataframe(
            self.test_df_feat, batch_size=self.batch_size_test,
            x_col='img_path', y_col='features_clean', target_size=im_dims,
            shuffle=False)

        return gen_train_cat, gen_test_cat, \
               gen_train_feat, gen_test_feat


    def get_image_dfs(self):
        '''
        Method to get the images, resize them and split them into
        training and test data.
        '''

        # If sample size not 'all', only get the first n images.
        if not self.sample_size == 'all':
            df_tmp = self.meta_data.iloc[:self.sample_size]
        else:
            df_tmp = self.meta_data.copy()

        df_tmp['img_path'] = df_tmp['hierarchy_clean'].apply(lambda x:
            self.path_images + str(Path(x).parents[1]) + '/' +
            x.split('/')[-1] + '.png')

        # Add a column with image classification.
        df_tmp['img_class'] = df_tmp['hierarchy_clean'].apply(
            os.path.dirname).str.replace('/', '_')

        # Only select the needed columns.
        df_sel_cat = df_tmp[['img_class', 'img_path']]
        df_sel_feat = df_tmp[['features_clean', 'img_path']]

        # Get train-test split.
        train_df_cat, test_df_cat = self.tt_split_cat(df_sel_cat,
                                                      'img_class')
        train_df_feat, test_df_feat = self.tt_split_feat(df_sel_feat,
                                                         'features_clean')

        return train_df_cat, test_df_cat, \
               train_df_feat, test_df_feat


    def tt_split_feat(self, df, col):
        '''
        Method that does a manual train-test split since the sklearn
        train-test split does not make sure that all features are included
        in the train and test set.
        '''

        # Convert string of list to proper list.
        df[col] = df[col].apply(literal_eval)

        # Get list of unique features.
        lst_uq = list(set([i for s in df[col].tolist() for i in s]))

        # Set empty lists.
        lst_train = []
        lst_test = []

        # Get a working copy of the data frame.
        df_w = df.copy()

        # Loop over the unique features.
        for f in lst_uq:

            # Get the temporary data frame of rows that contain the feature.
            mask = df_w[col].apply(lambda x: f in x)
            df_tmp = df_w[mask]

            # Get the train-test indexes for the temporary data frame.
            tr_idx, te_idx = self.tt_idx(df_tmp)

            # Add it to the existing lists.
            lst_train = lst_train + tr_idx
            lst_test = lst_test + te_idx

            # Exclude the rows that were already assigned.
            df_w = df_w[~df_w.index.isin(tr_idx)]
            df_w = df_w[~df_w.index.isin(te_idx)]

        # Make both lists unique.
        lst_train_uq = list(set(lst_train))
        lst_test_uq = list(set(lst_test))

        # Use the unique lists to select the data frames.
        df_train = df[df.index.isin(lst_train_uq)]
        df_test = df[df.index.isin(lst_test_uq)]

        return df_train, df_test


    def tt_split_cat(self, df, col):
        '''
        Method that does a manual train-test split since the sklearn
        train-test split does not make sure that all categories are included
        in the train and test set.
        '''

        # Get a list of unique classes.
        lst_uq = df[col].unique()

        # Set empty lists.
        lst_train = []
        lst_test = []

        # Loop through every item in unique list.
        for c in lst_uq:

            # Define temporary data frame.
            df_tmp = df[df[col] == c]

            # Get the train-test indexes for the temporary data frame.
            tr_idx, te_idx = self.tt_idx(df_tmp)

            # Add it to the existing lists.
            lst_train = lst_train + tr_idx
            lst_test = lst_test + te_idx

        # Use the lists to select the data frames.
        df_train = df[df.index.isin(lst_train)]
        df_test = df[df.index.isin(lst_test)]

        return df_train, df_test


    def tt_idx(self, df_tmp):
        '''
        Method that returns list of indexes for train and test data for the
        respective temporary index.
        '''

        if df_tmp.shape[0] > 5:

            # Set seed.
            np.random.seed(self.rand_state)

            # Generate random 0 and 1 in the length of cont.
            split_train = list(np.random.rand(df_tmp.shape[0]) >=
                               self.test_size)
            split_test = [not i for i in split_train]

            # Get the train list.
            lst_tr = list(compress(df_tmp.index, split_train))

            # Get the test list.
            lst_te = list(compress(df_tmp.index, split_test))

            return lst_tr, lst_te

        elif df_tmp.shape[0] <= 5 and df_tmp.shape[0] > 1:

            # Define ratio.
            rat = math.ceil(df_tmp.shape[0] / 2.0)

            # Get the train list.
            lst_tr = list(np.random.choice(list(df_tmp.index), rat,
                                           replace=False))

            # Get the test list.
            lst_te = [i for i in list(df_tmp.index) if i not in
                      lst_tr]

            return lst_tr, lst_te

        else:
            return list(df_tmp.index), list(df_tmp.index)


    def plot_result(self, res):
        '''
        Method to plot the accuracy and the loss of the training steps.
        '''

        # Get the accuracy and the loss.
        acc = res.history[self.metrics[0]]
        loss = res.history['loss']

        # Get the accuracy and the loss for the validation-set.
        acc_val = res.history[['val_' + v for v in self.metrics][0]]
        loss_val = res.history['val_loss']

        # Plot.
        plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
        plt.plot(loss, 'o', color='b', label='Training Loss')
        plt.plot(acc_val, linestyle='--', color='r', label='Validation Acc.')
        plt.plot(loss_val, 'o', color='r', label='Validation Loss')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Save.
        plt.savefig('./output_hier/accuracy_plot_{}.png'.format(
            time.strftime("%Y%m%d-%H%M%S")))

        # Close figure.
        plt.close()

        return


    def save_accuracy(self, res):
        '''
        Method to get the accuracy and loss and write it into a text file.
        '''

        # Get the accuracy and loss.
        result = res.evaluate_generator(self.test_gen,
                                        steps=self.steps_test())

        # Define a filename.
        f_name = './output_hier/accuracy_{}.txt'.format(time.strftime(
            "%Y%m%d-%H%M%S"))

        # Open file.
        f = open(f_name, 'w+')

        # Write content.
        f.write('{}: {}\r\n'.format(res.metrics_names[0], result[0]))
        f.write('{}: {}%\r\n'.format(res.metrics_names[1], result[1]))

        # Close file.
        f.close()

        return


    def save_model(self, res):
        '''
        Method to save the model into JSON and the weight into HDF5 files.
        '''

        # Define JSON and HDF5 file names.
        f_mod = './output_hier/model_{}.json'.format(time.strftime(
            "%Y%m%d-%H%M%S"))
        f_wgt = './output_hier/weights_{}.h5'.format(time.strftime(
            "%Y%m%d-%H%M%S"))

        # Save model.
        with open(f_mod, "w") as json_file:
            json_file.write(res.to_json())

        # Save weights.
        res.save_weights(f_wgt)

        return


if __name__ == '__main__':

    # Get the training done.
    cnn = GlobusCNN()