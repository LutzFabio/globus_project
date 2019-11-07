import pandas as pd
import numpy as np
import os
import cv2
import glob
import time
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
  GaussianNoise
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, \
  Adamax, Nadam
from tensorflow.keras.models import model_from_json


class HierarchicalCNN:


    def __init__(self, set, load=False):

        # Unbundle the settings.
        for s in set.keys():
            setattr(self, s, set[s])

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
                self.model = ResNet50(weights='imagenet')

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

        # Augment the pictures.
        self.augment_images()

        # Create new model.
        model_new = self.create_new_model()

        # Compile the model.
        model_new.compile(optimizer=self.optimizer,
                          loss=self.loss,
                          metrics=self.metrics)

        # Train the model.
        hist = model_new.fit_generator(generator=self.train_gen,
                                       epochs=self.epochs,
                                       steps_per_epoch=self.steps_per_epoch,
                                       class_weight=self.class_wgt(),
                                       validation_data=self.test_gen,
                                       validation_steps=self.steps_test())

        # Set the trained model as attribute.
        setattr(self, 'model_trained', hist)

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


    def class_wgt(self):
        '''
        Method to calculate the class weights, such that imbalanced classes
        could be handled.
        '''

        wgt = compute_class_weight(class_weight='balanced',
                                   classes=np.unique(self.label_bin.classes_),
                                   y=self.label_bin.classes_)

        return wgt


    def create_new_model(self):
        '''
        Method that creates a new model based on an existing one
        (transfer learning)
        '''

        # Initiate new model.
        new_model = Sequential()

        # Load pre-trained model.
        conv_model = Model(inputs=self.model.input,
                           outputs=self.model.get_layer(
                                   self.layer_old).output)

        # Set the layers to "not-trainable".
        conv_model.trainable = False

        # Check if something should be re-trained.
        if self.layer_old_train is not None:
            for layer in conv_model.layers:
                trainable = (self.layer_old_train in layer.name)
                layer.trainable = trainable

        # Add the pre-trained to the new model.
        new_model.add(conv_model)

        # Add additional layers.
        new_model.add(Flatten())
        #new_model.add(Dense(self.size_dense,
        #                    activation=self.activation_dense))
        #new_model.add(Dropout(self.dropout))
        new_model.add(Dense(len(self.label_bin.classes_),
                                activation=self.activation_last))

        # Add new model as attribute.
        setattr(self, 'model_new', new_model)

        return new_model


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
        self.train_df, self.test_df = self.get_image_dfs()

        # Initiate the train ImageDataGenerator.
        gen_train_data = ImageDataGenerator(
            rescale=self.im_resc,
            rotation_range=self.rot_range,
            width_shift_range=self.width_shift,
            height_shift_range=self.height_shift,
            shear_range=self.shear_rng,
            zoom_range=self.zoom_rng,
            horizontal_flip=self.hor_flip,
            vertical_flip=self.vert_flip,
            fill_mode=self.fill_mod)

        # Inititate the train generator.
        gen_train = gen_train_data.flow_from_dataframe(
            self.train_df, batch_size=self.batch_size_train,
            x_col='img_path', y_col='img_class', target_size=im_dims,
            shuffle=True)
            #save_to_dir=self.path_images + 'augmented/',
            #save_format='png')

        # Initiate the test ImageDataGenerator.
        gen_test_data = ImageDataGenerator(
            rescale=self.im_resc)

        # Inititate the train generator.
        gen_test = gen_test_data.flow_from_dataframe(
            self.test_df, batch_size=self.batch_size_test,
            x_col='img_path', y_col='img_class', target_size=im_dims,
            shuffle=False)

        # Set as properties.
        self.train_gen = gen_train
        self.test_gen = gen_test

        return


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

        # Add a column with full path.
        df_tmp['img_path'] = self.path_images + df_tmp['hierarchy_clean'] + \
                             '.png'

        # Add a column with image classification.
        df_tmp['img_class'] = df_tmp['hierarchy_clean'].apply(
            os.path.dirname).str.replace('/', '_')

        # Only select these two columns.
        df_sel = df_tmp[['img_class', 'img_path']]

        # # Create empty lists.
        # images = []
        # labels = []
        #
        # # Loop through all the images.
        # for i, row in df_tmp.iterrows():
        #
        #     # Create path to picture.
        #     # path_im = self.path_images + row['hierarchy_clean'] + '.webp'
        #     path_im = glob.glob(self.path_images + '**/' + str(
        #         row['globus_id']) + '.png', recursive=True)[0]
        #
        #     # Load image.
        #     im = cv2.imread(path_im)
        #
        #     # Resize the image.
        #     im_r = cv2.resize(im, dims)
        #
        #     # Image to array,
        #     im_a = img_to_array(im_r)
        #
        #     # Append image and label.
        #     images.append(im_a)
        #     labels.append(os.path.dirname(row['hierarchy_clean']))
        #
        # # Get arrays.
        # im_arr = np.array(images, dtype='float')
        # la_arr = np.array(labels)
        #
        # # Binarize labels.
        # la_bin = LabelBinarizer().fit(la_arr)
        #
        # # Add binarized lables as attribute.
        # setattr(self, 'label_bin', la_bin)

        # Get train-test split.
        train_df, test_df = train_test_split(df_sel,
                                             test_size=self.test_size,
                                             random_state=self.rand_state)


        return train_df, test_df


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

    ###########################################################################
    # SETTINGS
    ###########################################################################

    set_dict = {

        'im_resc':              1.0 / 255,
        'rot_range':            180,
        'width_shift':          0.1,
        'height_shift':         0.1,
        'shear_rng':            0.1,
        'zoom_rng':             [0.9, 1.5],
        'hor_flip':             True,
        'vert_flip':            True,
        'fill_mod':             'nearest',
        'batch_size_train':     64,
        'batch_size_test':      16,
        'meta_file':            '/home/fabiolutz/propulsion/globus_project/'
                                'd_data_cleaning/meta_clean_red_train.csv',
        'path_images':          '/home/fabiolutz/propulsion/globus_project/'
                                'e_model/images_small_new/',
        # 'meta_file':            '/home/ubuntu/efs/meta_clean_red_train.csv',
        # 'path_images':          '/home/ubuntu/efs/images_small_new/',
        'sample_size':          'all', # or 'all'
        'test_size':            0.2,
        'rand_state':           200,
        'model_str':            'ResNet50',
        'layer_old':            'conv5_block3_out',
        'layer_old_train':      None, # or None
        'activation_dense':     'relu',
        'size_dense':           1024,
        'dropout':              0.2,
        'activation_last':      'softmax',
        'optimizer_str':        'Adam',
        'learning_rate':        0.01,
        'loss':                 'categorical_crossentropy',
        'metrics':              ['categorical_accuracy'],
        'epochs':               2,
        'steps_per_epoch':      2,

    }

    ###########################################################################

    # Get the training done.
    cnn = HierarchicalCNN(set_dict,
                          load=False)