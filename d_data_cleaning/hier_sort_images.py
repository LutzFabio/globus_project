import pandas as pd
import os
import glob
import shutil
from ast import literal_eval


def sort_and_save(df, dir_curr, dir_tar):

    # Set n.
    n = 1

    # Loop over pictures.
    for i, row in df.iterrows():

        # Create name of picture.
        name_tmp = str(row['globus_id']) + '.webp'

        try:

            # Get the path of the picture in the current folder.
            path_cur_tmp = glob.glob(dir_curr + '**/' + name_tmp,
                                     recursive=True)[0]

            # Get the target path.
            path_tar_tmp = dir_tar + row['hierarchy_clean'] + '.webp'

            # Check whether the directory exist. If not, create the directory.
            tar_dir = row['hierarchy_clean'].rsplit('/', 1)[0]
            if not os.path.isdir(dir_tar + tar_dir):
                os.makedirs(dir_tar + tar_dir)

            # Copy the file in the target directory.
            shutil.copy(path_cur_tmp, path_tar_tmp)

            # Print progress.
            print('{} of {}'.format(n, df.shape[0]))

        except:
            with open('missing.txt', 'a') as t:
                t.write(name_tmp)

        # Add one to n.
        n += 1

    return


if __name__ == '__main__':

    # Define path to the raw csv and path to the saving location.
    path_c = '/home/ubuntu/efs/'
    im_dir = '/home/ubuntu/efs/images_small/'
    im_tar = '/home/ubuntu/efs/images_final/'

    # Read the data.
    data = pd.read_csv(path_c + 'meta_clean_red.csv')
    data.set_index('unique_id', inplace=True)
    data['features_clean'] = data['features_clean'].apply(literal_eval)

    # Clean the data.
    sort_and_save(data, im_dir, im_tar)