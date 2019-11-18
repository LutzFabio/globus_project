import pandas as pd
import numpy as np
import time
from ast import literal_eval
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Flatten, Dropout, Conv2D, Input, Activation,\
    GlobalAveragePooling2D, BatchNormalization, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, \
  Adamax, Nadam
from keras.callbacks import ModelCheckpoint, CSVLogger


class GlobusCNN:
    '''
    Build a class object that handles all the tasks the CNN
    should be performing.
    '''

    # Settings.
    model_type =             'own' # 'r50' or 'own'. If 'own', use tf.keras.
    im_resc =                 None
    rot_range =               180
    width_shift =             0.0
    height_shift =            0.0
    shear_rng =               0.0
    zoom_rng =                0.0
    hor_flip =                True
    vert_flip =               True
    fill_mod =                'nearest'
    batch_size_train =        32
    batch_size_test =         16
    meta_file =               './../c_data_cleaning/meta_clean_red_train.csv'
    path_images =             './images_small_new/'
    output_dir =              './'
    sample_frac =             'all'
    test_size =               0.3
    rand_state=               200
    model_str =               'ResNet50'
    model_input =             (224, 224, 3)
    model_input_gen =         (224, 224)
    num_layers_not_trained =  143
    size_dense =              1024
    dropout =                 0.2
    activation_mid_both =     'relu'
    activation_last_cat =     'softmax'
    activation_last_feat =    'sigmoid'
    optimizer_str =           'Adam'
    learning_rate =           0.001
    loss =                    {'category': 'categorical_crossentropy',
                               'feature': 'binary_crossentropy'}
    metrics =                 {'category':'accuracy',
                               'feature':'accuracy'}
    epochs =                  30
    batch_steps =             None
    validation_freq =         1
    mod_saved_after_epochs =  5


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

        # Initiate the Checkpoint logger that saves the model after ever
        # fifth epoch.
        self.cp_logger = ModelCheckpoint(
            self.output_dir + 'model_r50_{epoch:08d}.h5',
            save_weights_only=False,  period=self.mod_saved_after_epochs)

        # Define whether to train or to load a model and evaluate it.
        if not load:
            # Define the model to train.
            if self.model_str == 'ResNet50':
                self.base_model = ResNet50(weights='imagenet',
                                           include_top=False,
                                           input_shape=self.model_input)

            else:
                raise ValueError('{} not implemented yet!'.format(
                    self.model_str))

        else:
            raise ValueError('No loading methodology implemented yet!')


    def create_train_evaluate(self):
        '''
        Method that summarizes the different steps from creation over
        training until evaluation of a model.
        '''

        # Get the train and evaluation data frames.
        self.get_image_dfs()

        # Create the ImageDataGenerators used for training and evaluating the
        # model.
        self.create_generators()

        # Create the model, depending on which one was specified.
        if self.model_type == 'r50':

            self.create_r50_model()

        elif self.model_type == 'own':

            self.create_own_model()

        else:
            raise ValueError('Model "{}" not yet implemented!'.format(
                self.model_type))

        # Train the model.
        self.train()

        # Evaluate the model (incl. saving the model and other outputs).
        self.evaluate()

        return


    def train(self):
        '''
        Method that trains a new model and takes care of the
        tasks needed in this regard.
        '''

        # Compile the model.
        self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)

        # Train the model.
        self.model_trained = self.model.fit_generator(
            generator=self.train_gen, epochs=self.epochs,
            validation_data=self.test_gen,
            steps_per_epoch=self.steps_train(),
            validation_steps=self.steps_test(),
            callbacks=[self.csv_logger, self.cp_logger],
            validation_freq=self.validation_freq)

        return


    def evaluate(self):
        '''
        Method to evaluate the trained model.
        '''

        # Save model.
        self.save_model()

        # Save the predictions and the confusion matrices.
        self.save_evaluations()

        return


    def steps_train(self):
        '''
        Method to calculate the test steps.
        '''

        # Check if a number of batch steps is set. If not, calculate the
        # number.
        if self.batch_steps is None:
            bs = self.gen_pseudo_train_cat.n / self.batch_size_train
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
            bs = self.gen_pseudo_test_cat.n / self.batch_size_test
        else:
            bs = self.batch_steps

        return bs


    def create_r50_model(self):
        '''
        Method that creates a new model based on the pre-trained ResNet50
        model (transfer learning)
        '''

        # Set all the layers in the original model to non-trainable.
        self.base_model.trainable = True

        # Check if something should NOT be re-trained in the original ResNet50
        # model.
        if self.num_layers_not_trained is not None:
            if self.num_layers_not_trained != 'all':
                for layer in self.base_model.layers[
                             :self.num_layers_not_trained]:
                    layer.trainable = False
            else:
                for layer in self.base_model.layers:
                    layer.trainable = False

        # Add an input as well as a global average pooling layer to the
        # original model.
        inp = Input(self.model_input)
        mod = self.base_model(inp)
        mod = GlobalAveragePooling2D()(mod)

        # Create the categorical branch.
        mod_cat = Dense(len(self.gen_pseudo_train_cat.class_indices),
                        activation=self.activation_mid_both,
                        name='category_dense')(mod)
        mod_cat = Dropout(0.5)(mod_cat)
        mod_cat = Dense(len(self.gen_pseudo_train_cat.class_indices),
                        activation=self.activation_last_cat,
                        name='category')(mod_cat)

        # Create the feature branch.
        mod_feat = Dense(len(self.gen_pseudo_train_feat.class_indices),
                         activation=self.activation_mid_both,
                         name='feature_dense')(mod)
        mod_feat = Dropout(0.5)(mod_feat)
        mod_feat = Dense(len(self.gen_pseudo_train_feat.class_indices),
                         activation=self.activation_last_feat,
                         name='feature')(mod_feat)

        # Combine both branches to one model.
        self.model = Model(inputs=inp,
                           outputs=[mod_cat, mod_feat])

        return


    def create_own_model(self):
        '''
        Method to create an own model, i.e. without transfer learning.
        '''

        # Define an input layer.
        inputs = Input(self.model_input)

        # Create the categorical branch.
        cat = Conv2D(32, (3, 3), padding="same")(inputs)
        cat = Activation("relu")(cat)
        cat = BatchNormalization(axis=-1)(cat)
        cat = MaxPooling2D(pool_size=(3, 3))(cat)
        cat = Dropout(0.25)(cat)

        cat = Conv2D(64, (3, 3), padding="same")(cat)
        cat = Activation("relu")(cat)
        cat = BatchNormalization(axis=-1)(cat)
        cat = Conv2D(64, (3, 3), padding="same")(cat)
        cat = Activation("relu")(cat)
        cat = BatchNormalization(axis=-1)(cat)
        cat = MaxPooling2D(pool_size=(2, 2))(cat)
        cat = Dropout(0.25)(cat)

        cat = Flatten()(cat)
        cat = Dense(256)(cat)
        cat = Activation("relu")(cat)
        cat = BatchNormalization()(cat)
        cat = Dropout(0.5)(cat)
        cat = Dense(len(self.gen_pseudo_train_cat.class_indices))(cat)
        cat = Activation(self.activation_last_cat,
                         name="category")(cat)

        # Create the feature branch.
        feat = Conv2D(32, (3, 3), padding="same")(inputs)
        feat = Activation("relu")(feat)
        feat = BatchNormalization(axis=-1)(feat)
        feat = MaxPooling2D(pool_size=(3, 3))(feat)
        feat = Dropout(0.25)(feat)

        feat = Conv2D(64, (3, 3), padding="same")(feat)
        feat = Activation("relu")(feat)
        feat = BatchNormalization(axis=-1)(feat)
        feat = Conv2D(64, (3, 3), padding="same")(feat)
        feat = Activation("relu")(feat)
        feat = BatchNormalization(axis=-1)(feat)
        feat = MaxPooling2D(pool_size=(2, 2))(feat)
        feat = Dropout(0.25)(feat)

        feat = Flatten()(feat)
        feat = Dense(256)(feat)
        feat = Activation("relu")(feat)
        feat = BatchNormalization()(feat)
        feat = Dropout(0.5)(feat)
        feat = Dense(len(self.gen_pseudo_train_feat.class_indices))(feat)
        feat = Activation(self.activation_last_feat,
                          name="feature")(feat)

        # Combine both branches to one model.
        self.model = Model(inputs=inputs,
                           outputs=[cat, feat])

        return


    def create_generators(self):
        '''
        Method that creates the ImageDataGenerators and everything that is
        needed in this regard.
        '''

        # Initiate the train and test ImageDataGenerator.
        train_gen_init = ImageDataGenerator(
            preprocessing_function=preprocess_input,
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
            preprocessing_function=preprocess_input,
            rescale=self.im_resc)

        # Construct the train and test generators for multiple outputs.
        self.train_gen = self.combine_generators(train_gen_init,
                                                 self.train_df,
                                                 self.batch_size_train)
        self.test_gen = self.combine_generators(test_gen_init,
                                                self.test_df,
                                                self.batch_size_test)

        # Create a test generator for prediction. This is needed because the
        # custom generators cannot be reset.
        self.test_gen_pred = self.combine_generators(test_gen_init,
                                                     self.test_df,
                                                     self.batch_size_test)

        # Due to the custom generator constructed above, the following
        # function generates some attributes and save some data to CSV's
        # that are needed but cannot be derived from the custom generator
        # anymore.
        self.create_pseudo_generators(train_gen_init, test_gen_init)

        return


    def create_pseudo_generators(self, train_gen_init, test_gen_init):
        '''
        Method that generates some attributes that are needed later. This is
        necessary because a custom ImageDataGenerator had to be created.
        '''

        # Generate the two pseude train and test generators.
        self.gen_pseudo_train_cat = train_gen_init.flow_from_dataframe(
            self.train_df, batch_size=self.batch_size_train,
            x_col='img_path', y_col='img_class',
            target_size=self.model_input_gen, shuffle=False,
            class_mode='categorical', seed=self.rand_state)

        self.gen_pseudo_train_feat = train_gen_init.flow_from_dataframe(
            self.train_df, batch_size=self.batch_size_train,
            x_col='img_path', y_col='features_clean',
            target_size=self.model_input_gen, shuffle=False,
            class_mode='categorical', seed=self.rand_state)

        self.gen_pseudo_test_cat = test_gen_init.flow_from_dataframe(
            self.test_df, batch_size=self.batch_size_test,
            x_col='img_path', y_col='img_class',
            target_size=self.model_input_gen, shuffle=False,
            class_mode='categorical', seed=self.rand_state)

        self.gen_pseudo_test_feat = test_gen_init.flow_from_dataframe(
            self.test_df, batch_size=self.batch_size_test,
            x_col='img_path', y_col='features_clean',
            target_size=self.model_input_gen, shuffle=False,
            class_mode='categorical', seed=self.rand_state)

        return


    def combine_generators(self, gen_init, df, batch_size):
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
            df, batch_size=batch_size, x_col='img_path', y_col='img_class',
            target_size=im_dims_tup, shuffle=False, class_mode='categorical',
            seed=self.rand_state)

        # Construct the feature generator.
        self.gen_feat = gen_init.flow_from_dataframe(
            df, batch_size=batch_size, x_col='img_path',
            y_col='features_clean', target_size=im_dims_tup, shuffle=False,
            class_mode='categorical', seed=self.rand_state)

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
        self.train_df, self.test_df = self.tt_split(df_strat)

        return


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


    def save_evaluations(self):
        '''
        Method to save the predictions and confusion matrices to CSV files.
        '''

        # Get the predictions.
        pred = self.model.predict_generator(self.test_gen_pred,
                                            steps=self.steps_test())

        # Get the confusion matrix for the categories and convert it into a
        # data frame.
        conf_mat_raw_cat = confusion_matrix(self.gen_pseudo_test_cat.classes,
                                            np.argmax(pred[0], axis=1))
        conf_mat_cat = pd.DataFrame(data=conf_mat_raw_cat, columns=list(
            self.gen_pseudo_test_cat.class_indices.keys()), index=list(
            self.gen_pseudo_test_cat.class_indices.keys()))

        # Since there does not exist a confusion matrix for multi-label
        # predictions, a special type of data frame is constructed that is
        # somewhat similar to a confusion matrix. To start, the classes are
        # encoded and the predictions are made to be at one or zero.
        classes_feat_trafo = MultiLabelBinarizer().fit_transform(
            self.gen_pseudo_test_feat.classes)
        predict_feat_trafo = pred[1]
        predict_feat_trafo[predict_feat_trafo >= 0.5] = 1
        predict_feat_trafo[predict_feat_trafo < 0.5] = 0

        # A loop is applied that loops over all features and does a
        # confusion matrix for each feature in terms of whether it was
        # predicted or not. The result is then transformed into a data frame
        # and all the small dataframes are concatenated to a single large
        # data frame.
        conf_mat_feat = pd.DataFrame()
        for e in range(0, predict_feat_trafo.shape[1]):
            mat_tmp = confusion_matrix(classes_feat_trafo[:, e],
                                       predict_feat_trafo[:, e])
            lbl = list(self.gen_pseudo_test_feat.class_indices.keys())[e]
            lbl_alt = lbl + '_not'
            df_tmp = pd.DataFrame(data=mat_tmp, columns=[lbl, lbl_alt],
                                  index=[lbl, lbl_alt])
            conf_mat_feat = pd.concat([conf_mat_feat, df_tmp])

        # Save the two data frames into CSVs.
        conf_mat_cat.to_csv(
            self.output_dir + 'conf_mat_cat_{}_{}.csv'.format(self.model_type,
                time.strftime("%Y%m%d-%H%M%S")))
        conf_mat_feat.to_csv(
            self.output_dir + 'conf_mat_feat_{}_{}.csv'.format(self.model_type,
                time.strftime("%Y%m%d-%H%M%S")))

        # Next, the predictions for every test image will also be saved into
        # CSV files.
        pred_cat_df = pd.DataFrame(data=pred[0], columns=list(
            self.gen_pseudo_test_cat.class_indices.keys()),
            index=self.gen_pseudo_test_cat.filenames)
        pred_feat_df = pd.DataFrame(data=pred[1], columns=list(
            self.gen_pseudo_train_feat.class_indices.keys()),
            index=self.gen_pseudo_test_feat.filenames)

        idx = self.test_df[self.test_df['img_path'].isin(
            pred_cat_df.index)].index

        pred_cat_df['true_cat'] = self.test_df['img_class'].loc[idx].values
        pred_feat_df['true_cat'] = self.test_df['features_clean'].loc[
            idx].values

        # Save the two data frames into CSVs.
        pred_cat_df.to_csv(
            self.output_dir + 'pred_cat_{}_{}.csv'.format(self.model_type,
                time.strftime("%Y%m%d-%H%M%S")))
        pred_feat_df.to_csv(
            self.output_dir + 'pred_feat_{}_{}.csv'.format(self.model_type,
                time.strftime("%Y%m%d-%H%M%S")))

        return


    def save_model(self):
        '''
        Method to save the model into H5 files and to save other outputs
        into CSV files.
        '''

        # Define H5 file name.
        f_wgt = self.output_dir + 'model_{}_{}.h5'.format(
            self.model_type, time.strftime("%Y%m%d-%H%M%S"))

        # Save model.
        self.model.save_weights(f_wgt)

        # Get data frames with the categories and features, according to the
        # encoding in the generator.
        df_cat = pd.DataFrame.from_dict(
            self.gen_pseudo_train_cat.class_indices,
            orient='index', columns=['encoding']).reset_index().rename(
            columns={'index': 'categories'})

        df_feat = pd.DataFrame.from_dict(
            self.gen_pseudo_train_feat.class_indices,
            orient='index', columns=['encoding']).reset_index().rename(
            columns={'index': 'features'})

        # Save the data frames to a CSV such that it can later be used in
        # prediction.
        df_cat.to_csv(self.output_dir + 'categories_{}_{}.csv'.format(
            self.model_type, time.strftime("%Y%m%d-%H%M%S")))

        df_feat.to_csv(self.output_dir + 'features_{}_{}.csv'.format(
            self.model_type, time.strftime("%Y%m%d-%H%M%S")))

        return


if __name__ == '__main__':

    # Get the training done.
    cnn = GlobusCNN()
    cnn.create_train_evaluate()