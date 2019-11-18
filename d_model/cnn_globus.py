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
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, \
    decode_predictions
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, \
  GaussianNoise, Input, Activation, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, \
  Adamax, Nadam
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


class GlobusCNN:
    '''
    Build a class object that handles all the tasks the CNN
    should be performing.
    '''

    # Settings.
    im_resc =              1.0 / 255
    rot_range =            180
    width_shift =          0.1
    height_shift =         0.1
    shear_rng =            0.1
    zoom_rng =             [0.9, 1.5]
    hor_flip =             True
    vert_flip =            True
    fill_mod =             'nearest'
    batch_size_train =     16
    batch_size_test =      8
    meta_file =            '/home/fabiolutz/propulsion/globus_project/' \
                           'd_data_cleaning/meta_clean_red_train.csv'
    path_images =          '/home/fabiolutz/propulsion/globus_project/' \
                           'e_model/images_small_new/'
    # meta_file =            '/home/ubuntu/efs/meta_clean_red_train.csv'
    # path_images =          '/home/ubuntu/efs/images_small_new/'
    output_dir =           './output_glob_test/'
    sample_frac =          'all'
    test_size =            0.3
    rand_state=            200
    model_str =            'ResNet50'
    model_input=           (224, 224, 3)
    layer_old =            'conv5_block3_out'
    layer_old_train =      ['conv5_block2', 'conv5_block3']
    activation_dense =     'relu'
    size_dense =           1024
    size_output =          1
    dropout =              0.2
    activation_last_cat =  'softmax'
    activation_last_feat = 'sigmoid'
    optimizer_str =        'Adam'
    learning_rate =        0.05
    loss =                 {'category': 'binary_crossentropy',
                            'feature': 'binary_crossentropy'}
    metrics =              ['accuracy']
    epochs =               30
    batch_steps =          None
    validation_freq =      3


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

        # Initiate the CSV logger.
        self.csv_logger = CSVLogger(self.output_dir + 'training_{}.log'.format(
            time.strftime("%Y%m%d-%H%M%S")), separator=',', append=False)

        # Initiate the Checkpoint logger.
        self.cat_cp_logger = ModelCheckpoint(
            self.output_dir + 'best_cat_model.hdf5', save_best_only=True,
            monitor='val_category_' + self.metrics[0], mode='max')

        self.feat_cp_logger = ModelCheckpoint(
            self.output_dir + 'best_feat_model.hdf5', save_best_only=True,
            monitor='val_feature_' + self.metrics[0], mode='max')

        # Define whether to train or to load a model and evaluate it.
        if not load:
            # Define the model to train.
            if self.model_str == 'ResNet50':
                self.model = ResNet50(weights='imagenet', include_top=False,
                                      input_shape=self.model_input)
            #
            # elif self.model_str == 'VGG16':
                # mod_tmp = VGG16(weights='imagenet')
                # last_layer = mod_tmp.get_layer('block5_pool')
                # self.model = Sequential().add(Model(inputs=self.model_input,
                #                                     outputs=last_layer))

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
            self.train_gen, self.test_gen = self.create_generators()

            # Compile the model.
            loaded_model.compile(optimizer=self.optimizer,
                                 loss=self.loss,
                                 metrics=self.metrics)
            self.model_loaded = loaded_model
            self.model_loaded_hist = pd.read_csv(self.latest_log())

            # Evaluate.
            self.evaluate_load()


    def train(self):
        '''
        Method that trains a new model and takes care of the
        tasks needed in this regard.
        '''

        # Augment the pictures.
        self.train_gen, self.test_gen = self.create_generators()

        # Create extended model.
        self.model_ext = self.create_new_model()

        # Compile the model.
        self.model_ext.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)

        # Train the model.
        self.model_trained = self.model_ext.fit_generator(
            generator=self.train_gen, epochs=self.epochs,
            validation_data=self.test_gen,
            steps_per_epoch=self.steps_train(),
            validation_steps=self.steps_test(),
            callbacks=[self.csv_logger, self.cat_cp_logger,
                       self.feat_cp_logger],
            use_multiprocessing=True,
            validation_freq=self.validation_freq)

        return


    def evaluate_train(self):
        '''
        Method to evaluate the trained model.
        '''

        # Save model.
        self.save_model_train()

        # Plot and save an image of the accuracy to the output folder.
        self.plot_result_train()

        # Save the accuracy in a text file.
        self.save_accuracy_train()

        return


    def evaluate_load(self):
        '''
        Method to evaluate the loaded model.
        '''

        # Save the accuracy in a text file.
        self.save_accuracy_load()

        # Plot and save an image of the accuracy to the output folder.
        self.plot_result_load()

        return


    def latest_h5(self):
        '''
        Method to get the path of the latest HDF5 weight file.
        '''

        # Get latest HDF5 file from output folder.
        lst_files = glob.glob(self.output_dir + '*.h5')
        latest = max(lst_files, key=os.path.getctime)

        return latest


    def latest_log(self):
        '''
        Method to get the path of the latest logger file.
        '''

        # Get latest logger file from output folder.
        lst_files = glob.glob(self.output_dir + '*.log')
        latest = max(lst_files, key=os.path.getctime)

        return latest


    def load_json_model(self):
        '''
        Method to load the JSON model.
        '''

        # Get latest JSON file from output folder.
        lst_files = glob.glob(self.output_dir + '*.json')
        latest = max(lst_files, key=os.path.getctime)

        # Open, assign and close the JSON file.
        j_file = open(latest, 'r')
        j_model = j_file.read()
        j_file.close()

        return j_model


    def steps_train(self):
        '''
        Method to calculate the test steps.
        '''

        # Check if a number of batch steps is set. If not, calculate the
        # number.
        if self.batch_steps is None:
            bs = self.len_train_df / self.batch_size_train
        else:
            bs = self.batch_steps

        return bs


    def steps_test(self):
        '''
        Method to calculate the test steps.
        '''

        # Check if a number of batch steps is set. If not, calculate the
        # number.
        if self.batch_steps is None:
            bs = self.len_test_df / self.batch_size_test
        else:
            bs = self.batch_steps

        return bs


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
                trainable = (self.layer_old_train[0] in layer.name or
                             self.layer_old_train[1] in layer.name)
                layer.trainable = trainable

        # Create the categorical model.
        model_cat = self.model.outputs[0]
        model_cat = GlobalAveragePooling2D()(model_cat)
        model_cat = Dense(self.len_cat_uq,
                          activation=self.activation_dense)(model_cat)
        model_cat = Activation(self.activation_last_cat,
                               name='category')(model_cat)

        # Create the feature model.
        model_feat = self.model.outputs[0]
        model_feat = GlobalAveragePooling2D()(model_feat)
        model_feat = Dense(self.len_feat_uq,
                           activation=self.activation_dense)(model_feat)
        model_feat = Activation(self.activation_last_feat,
                                name='feature')(model_feat)

        # Combine the model.
        model = Model(inputs=self.model.input,
                      outputs=[model_cat, model_feat])

        return model


    def create_generators(self):
        '''
        Method that creates the ImageDataGenerators and everything that is
        needed in this regard.
        '''

        # Get images for train and test.
        self.train_df, self.test_df = self.get_image_dfs()

        # Initiate the train and test ImageDataGenerator.
        train_gen_init = ImageDataGenerator(
            rescale=self.im_resc,
            rotation_range=self.rot_range,
            width_shift_range=self.width_shift,
            height_shift_range=self.height_shift,
            shear_range=self.shear_rng,
            zoom_range=self.zoom_rng,
            horizontal_flip=self.hor_flip,
            vertical_flip=self.vert_flip,
            fill_mode=self.fill_mod)

        test_gen_init = ImageDataGenerator(
            rescale=self.im_resc)

        # Construct the train and test generators.
        train_gen = self.combine_generators(train_gen_init, self.train_df)
        test_gen = self.combine_generators(test_gen_init, self.test_df)

        # Define unique features.
        self.len_train_df = self.train_df.shape[0]
        self.len_test_df = self.test_df.shape[0]

        # Due to the custom generator constructed above, the following
        # function generates some attributes and save some data to CSV's
        # that are needed but cannot be derived from the custom generator
        # anymore.
        self.get_needed_attr_outp(train_gen_init)

        return train_gen, test_gen


    def get_needed_attr_outp(self, train_gen_init):
        '''
        Method that generates some attributes that are needed later. This is
        necessary because a custom ImageDataGenerator had to be created.
        '''

        # Generate the two train generators.
        gen_cat = train_gen_init.flow_from_dataframe(
            self.train_df, batch_size=self.batch_size_train,
            x_col='img_path', y_col='img_class', target_size=(224, 224),
            shuffle=False, class_mode='categorical', seed=self.rand_state)

        gen_feat = train_gen_init.flow_from_dataframe(
            self.train_df, batch_size=self.batch_size_train,
            x_col='img_path', y_col='features_clean', target_size=(224, 224),
            shuffle=False, class_mode='categorical', seed=self.rand_state)

        # Get the length of the class indices (unique categories and features).
        self.len_cat_uq = len(gen_cat.class_indices)
        self.len_feat_uq = len(gen_feat.class_indices)

        # Get data frames with the categories and features, according to the
        # encoding in the generator.
        df_cat = pd.DataFrame.from_dict(gen_cat.class_indices,
            orient='index', columns=['encoding']).reset_index().rename(
            columns={'index': 'categories'})

        df_feat = pd.DataFrame.from_dict(gen_feat.class_indices,
            orient='index', columns=['encoding']).reset_index().rename(
            columns={'index': 'features'})

        # Save the data frames to a CSV such that it can later be used in
        # prediction.
        df_cat.to_csv(self.output_dir + 'categories_{}.csv'.format(
            time.strftime("%Y%m%d-%H%M%S")))

        df_feat.to_csv(self.output_dir + 'features_{}.csv'.format(
            time.strftime("%Y%m%d-%H%M%S")))

        return


    def combine_generators(self, gen_init, df):
        '''
        Generator that is later fed into the 'fit_generator' method of the 
        network to be trained.

        Source: https://stackoverflow.com/questions/38972380/keras-how-to-
        use-fit-generator-with-multiple-outputs-of-different-type/41872896
        '''

        # Get image dimensions.
        try:
            im_dims_tup = self.model.layers[0].output_shape[0][1:3]
        except:
            im_dims_tup = self.model.layers[0].input_shape[1:3]

        # Construct the categorical generator.
        self.gen_cat = gen_init.flow_from_dataframe(
            df, batch_size=self.batch_size_train,
            x_col='img_path', y_col='img_class', target_size=im_dims_tup,
            shuffle=False, class_mode='categorical', seed=self.rand_state)

        # Construct the feature generator.
        self.gen_feat = gen_init.flow_from_dataframe(
            df, batch_size=self.batch_size_train,
            x_col='img_path', y_col='features_clean', target_size=im_dims_tup,
            shuffle=False, class_mode='categorical', seed=self.rand_state)

        # Build a while loop to construct the generator.
        while True:

            # Go to the next batch.
            x_tmp = next(self.gen_cat)
            y_tmp = next(self.gen_feat)

            # Assign the encoded arrays to variables that are then yielded.
            x = x_tmp[0]
            y_1 = x_tmp[1]
            y_2 = y_tmp[1]

            yield x, [y_1, y_2]


    def get_image_dfs(self):
        '''
        Method to get the images, resize them and split them into
        training and test data.
        '''

        # Make a working copy of the meta data frame.
        df_tmp = self.meta_data.copy()

        # Define the image path.
        df_tmp['img_path'] = df_tmp['hierarchy_clean'].apply(
            lambda x: self.path_images + str(Path(x).parents[1]) + '/' +
            x.split('/')[-1] + '.png')

        # Add a column with image classification.
        df_tmp['img_class'] = df_tmp['hierarchy_clean'].apply(
            os.path.dirname).str.replace('/', '_').str.replace('-', '')

        # Only select the needed columns.
        df_sel = df_tmp[['img_path', 'img_class', 'features_clean']]

        # Take a stratified sample, if specified.
        if not self.sample_frac == 'all':
            df_strat = df_sel.sample(frac=self.sample_frac,
                                     random_state=self.rand_state,
                                     replace=False)
        else:
            df_strat = df_sel.copy()

        # Get train and test data.
        train_df, test_df = self.tt_split(df_strat)

        return train_df, test_df


    def tt_split(self, df):

        # Convert string of list to proper list.
        df['features_clean'] = df['features_clean'].apply(literal_eval)

        # Get the unique classifications and features.
        lst_cls_uq = df['img_class'].unique()
        lst_feat_uq = list(set([i for s in df['features_clean'].tolist() for i
                                in s]))

        # Obtain the list lengths.
        len_lst_cls_uq = len(lst_cls_uq)
        len_lst_feat_uq = len(lst_feat_uq)

        # Do an initial train-test split.
        train_df, test_df = train_test_split(df, test_size=self.test_size)

        # Define the unique categories and features of the train and test
        # data frame.
        len_cls_uq_tr = len(train_df['img_class'].unique())
        len_feat_uq_tr = len(list(set([i for s in train_df[
            'features_clean'].tolist() for i in s])))

        len_cls_uq_te = len(test_df['img_class'].unique())
        len_feat_uq_te = len(list(set([i for s in test_df[
            'features_clean'].tolist() for i in s])))

        # Initiate a variable to handle the number of iterations.
        n_iter = 0

        # Make a while loops that splits the data frame into train and test
        # data as long as the classifications and/or features are missing in
        # either train or test data. The while loop is capped at 500
        # iterations in order to protect the loop for looping infinitely.
        while len_cls_uq_tr != len_lst_cls_uq or len_feat_uq_tr !=  \
                len_lst_feat_uq or len_cls_uq_te != len_lst_cls_uq or  \
                len_feat_uq_te != len_lst_feat_uq:

            # Make the loop break if the number of iterations is larger than
            # 500.
            if n_iter > 500:
                raise StopIteration('No valid train-test split found!')

            # Do a random train-test split without seed.
            train_df, test_df = train_test_split(df, test_size=self.test_size)

            # Define the unique categories and features of the train and
            # test data frame.
            len_cls_uq_tr = len(train_df['img_class'].unique())
            len_feat_uq_tr = len(list(set([i for s in train_df[
                'features_clean'].tolist() for i in s])))

            len_cls_uq_te = len(test_df['img_class'].unique())
            len_feat_uq_te = len(list(set([i for s in test_df[
                'features_clean'].tolist() for i in s])))

            # Add one to n.
            n_iter += 1

        print('Optimal train-test split found after {} '
              'iterations.'.format(n_iter))
        return train_df, test_df


    def plot_result_load(self):
        '''
        Method to plot the results from a loaded model.
        '''

        # Get the accuracy and the loss.
        cat_acc = self.model_loaded_hist['category_' + self.metrics[0]]
        feat_acc = self.model_loaded_hist['feature_' + self.metrics[0]]
        cat_loss = self.model_loaded_hist['category_loss']
        feat_loss = self.model_loaded_hist['feature_loss']

        # Get the accuracy and the loss for the validation-set.
        cat_val_acc = self.model_loaded_hist['val_category_' + self.metrics[0]]
        feat_val_acc = self.model_loaded_hist['val_feature_' + self.metrics[0]]
        cat_val_loss = self.model_loaded_hist['val_category_loss']
        feat_val_loss = self.model_loaded_hist['val_feature_loss']

        # Plot.
        plt.plot(cat_acc, linestyle='-', color='cornflowerblue',
                 label='Categorical training acc.')
        plt.plot(feat_acc, linestyle='-', color='limegreen',
                 label='Feature training acc.')
        plt.plot(cat_loss, linestyle='--', color='lightcoral',
                 label='Categorical training Loss')
        plt.plot(feat_loss, linestyle='--', color='sandybrown',
                 label='Feature training Loss')
        plt.plot(cat_val_acc, linestyle='-', marker='o', color='blue',
                 label='Categorical validation acc.')
        plt.plot(feat_val_acc, linestyle='-', marker='o', color='darkgreen',
                 label='Feature validation acc.')
        plt.plot(cat_val_loss, linestyle='--', marker='o', color='red',
                 label='Categorical validation loss')
        plt.plot(feat_val_loss, linestyle='--', marker='o', color='darkorange',
                 label='Feature validation loss')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Save.
        plt.savefig(self.output_dir + 'accuracy_plot_{}.png'.format(
            time.strftime("%Y%m%d-%H%M%S")))

        # Close figure.
        plt.close()

        return


    def plot_result_train(self):
        '''
        Method to plot the results of the trained model.
        '''

        # Get the accuracy and the loss.
        cat_acc = self.model_trained.history['category_' + self.metrics[0]]
        feat_acc = self.model_trained.history['feature_' + self.metrics[0]]
        cat_loss = self.model_trained.history['category_loss']
        feat_loss = self.model_trained.history['feature_loss']

        # Get the accuracy and the loss for the validation-set.
        cat_val_acc = self.model_trained.history['val_category_' +
                                                 self.metrics[0]]
        feat_val_acc = self.model_trained.history['val_feature_' +
                                                  self.metrics[0]]
        cat_val_loss = self.model_trained.history['val_category_loss']
        feat_val_loss = self.model_trained.history['val_feature_loss']

        # Plot.
        plt.plot(cat_acc, linestyle='-', color='cornflowerblue',
                 label='Categorical training acc.')
        plt.plot(feat_acc, linestyle='-', color='limegreen',
                 label='Feature training acc.')
        plt.plot(cat_loss, linestyle='--', color='lightcoral',
                 label='Categorical training Loss')
        plt.plot(feat_loss, linestyle='--', color='sandybrown',
                 label='Feature training Loss')
        plt.plot(cat_val_acc, linestyle='-', marker='o', color='blue',
                 label='Categorical validation acc.')
        plt.plot(feat_val_acc, linestyle='-', marker='o', color='darkgreen',
                 label='Categorical validation acc.')
        plt.plot(cat_val_loss, linestyle='--', marker='o', color='red',
                 label='Categorical validation loss')
        plt.plot(feat_val_loss, linestyle='--', marker='o', color='darkorange',
                 label='Feature validation loss')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        # Save.
        plt.savefig(self.output_dir + 'accuracy_plot_{}.png'.format(
            time.strftime("%Y%m%d-%H%M%S")))

        # Close figure.
        plt.close()

        return


    def save_accuracy_load(self):
        '''
        Method to get the accuracy and loss of a loaded model and write it
        into a text file.
        '''

        # Get the last row of the history and transpose it.
        df_last_row = self.model_loaded_hist.iloc[-1].T

        # Define a filename.
        f_name = self.output_dir + 'accuracy_{}.txt'.format(time.strftime(
            "%Y%m%d-%H%M%S"))

        # Open file.
        f = open(f_name, 'w+')

        # Write content.
        for i, r in df_last_row.iteritems():
            if 'accuracy' in i:
                f.write('{}: {}%\r\n'.format(i, r * 100))
            else:
                f.write('{}: {}\r\n'.format(i, r))

        # Close file.
        f.close()

        return


    def save_accuracy_train(self):
        '''
        Method to get the accuracy and loss of the trained model and write it
        into a text file.
        '''

        # Get the accuracy and loss.
        result = self.model_ext.evaluate_generator(self.test_gen,
                                                   steps=self.steps_test())

        # Define a filename.
        f_name = self.output_dir + 'accuracy_{}.txt'.format(time.strftime(
            "%Y%m%d-%H%M%S"))

        # Open file.
        f = open(f_name, 'w+')

        # Write content.
        f.write('{}: {}\r\n'.format(self.model_ext.metrics_names[0],
                                    result[0]))
        f.write('{}: {}%\r\n'.format(self.model_ext.metrics_names[1],
                                     result[1]))

        # Close file.
        f.close()

        return


    def save_model_train(self):
        '''
        Method to save the model into JSON and the weight into HDF5 files.
        '''

        # Define JSON and HDF5 file names.
        f_mod = self.output_dir + 'model_{}.json'.format(time.strftime(
            "%Y%m%d-%H%M%S"))
        f_wgt = self.output_dir + 'weights_{}.h5'.format(time.strftime(
            "%Y%m%d-%H%M%S"))

        # Save model.
        with open(f_mod, "w") as json_file:
            json_file.write(self.model_ext.to_json())

        # Save weights.
        self.model_ext.save_weights(f_wgt)

        return


if __name__ == '__main__':

    # Get the training done.
    cnn = GlobusCNN(load=False)