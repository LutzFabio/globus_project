import numpy as np
import os
import math
import shutil
from pathlib import Path
from itertools import compress
from PIL import Image
import webp
import warnings


def train_test(im_dir, tt_dir, tt_ratio):

    # Remove old pictures.
    if os.path.isdir(tt_dir):
        shutil.rmtree(tt_dir)

    # Get the directories with the files.
    tup_tmp = os.walk(im_dir)

    # Get the list of directories without the first one.
    lst_sub = [x[0] for x in tup_tmp]

    # Loop through the paths.
    for p in lst_sub:

        # Define path as Path object.
        pa = Path(p)

        # Get relative paths.
        pa_rel = str(pa.relative_to(im_dir)).replace('/', '_')

        # Get the directory content equal to webp format.
        cont = [i for i in os.listdir(p) if '.webp' in i]

        # If cont is empty, continue. Else split between train and test data.
        if cont:

            # Only do a train-test split for directories with more than four
            # pictures.
            if len(cont) > 5:

                # Generate random 0 and 1 in the length of cont.
                split_test = list(np.random.rand(len(cont)) >= tt_ratio)
                split_train = [not i for i in split_test]

                # Get the train list.
                lst_train = list(compress(cont, split_train))

                # Get the test list.
                lst_test = list(compress(cont, split_test))

            elif len(cont) <= 5 and len(cont) > 1:

                # Define ratio.
                rat = math.ceil(len(cont) / 2.0)

                # Get the train list.
                lst_train = list(np.random.choice(cont, rat, replace=False))

                # Get the test list.
                lst_test = [i for i in cont if i not in lst_train]

            else:
                # Assign all images to train and test.
                lst_train = cont
                lst_test = cont

            # Check if train folder already exists. If not, create it.
            if not os.path.isdir(tt_dir + 'train/' + pa_rel):
                os.makedirs(tt_dir + 'train/' + pa_rel)

            # Loop through the train list, copy the files and save them in
            # the train folder.
            for e in lst_train:
                im = webp.load_images(p + '/' + e, 'RGBA')[0]
                im.save(tt_dir + 'train/' + pa_rel + '/' + e.split('.')[0] +
                        '.png')
                # shutil.copy(p + '/' + e, tt_dir + 'train/' + pa_rel + '/' \
                    # + e)

            # Check if test folder already exists. If not, create it.
            if not os.path.isdir(tt_dir + 'test/' + pa_rel):
                os.makedirs(tt_dir + 'test/' + pa_rel)

            # Loop through the test list, copy the files and save them in
            # the test folder.
            for e in lst_test:
                im = webp.load_images(p + '/' + e, 'RGBA')[0]
                im.save(tt_dir + 'test/' + pa_rel + '/' + e.split('.')[0] +
                        '.png')
                # shutil.copy(p + '/' + e, tt_dir + 'test/' + pa_rel + '/' \
                # + e)

    return


if __name__ == '__main__':

    # Define paths.
    im_dir = '/home/fabiolutz/propulsion/globus_project/e_model/pictures_final/'
    tt_dir = '/home/fabiolutz/propulsion/globus_project/e_model/pictures_split/'

    # Train-test split.
    tt_ratio = 0.8

    train_test(im_dir, tt_dir, tt_ratio)