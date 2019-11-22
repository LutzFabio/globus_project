import pandas as pd
import numpy as np
import re
from ast import literal_eval
from collections import Counter
from pathlib import Path
import os


# Manual definition of categories to be included in training the model.
# These categories were selected since they belong to clothes, have proper
# data available and occur often enough. Thus, smaller categories were
# excluded but could be included in further development of the algorithm.
# For the features, this is handled by the function "listRareFeat" that
# returns a list of the features with less than 1'000 occurences, which are
# also excluded.
cat = ['taschen', 'schuhe', 'pullover-strick', 'top-shirts-sweats',
       'schals', 'hemden', 'hosen', 'oberteile', 'blusen-tuniken',
       'shirts-tops-sweats', 'kleider']

# The fit types belong often (or always) to the description. In order to
# have clean names, these following attributes are cleaned from the
# description and put to the features.
fit_types = ['slimfit','regularfit','skinnyfit','loosfit','taperedfit',
             'tailliert']

# Similarly to fit types, colors often have some patterns associated with
# them. In order to have clean colors, these patterns were cleaned and moved
# to the features too.
pat = ['gestreift', 'klein gemustert', 'kariert', 'karo', 'Glattleder',
       'Lack']

# The following cols are selected at the very end, since for training only a
# few columns are needed.
cols_sel = ['globus_id', 'descr_clean', 'hierarchy_clean',
            'gender', 'source_color', 'color_clean', 'season',
            'features_clean', 'img_class', 'img_path']


def data_clean(df, path_img):

    # Only select the categories, based on hierarchy_2.
    df_red = df[df['hierarchy_2'].isin(cat)]

    # Get rid of the entries with description of "others".
    df_red = df_red[df_red['descr'] != 'others']

    # Fill the gender gaps. Since only gender are missing for lingerie,
    # a simple "fillna" command can be used.
    df_red['gender'].fillna('damen', inplace=True)

    # Drop all elements without source_color.
    df_red = df_red[df_red['source_color'].notna()]

    # Fill a missing color with the source_color.
    df_red['color'].fillna(df_red['source_color'], inplace=True)

    # Get the pattern.
    df_red['pattern'] = df_red['color'].apply(patFromCol)

    # Get clean color.
    df_red['color_clean'] = df_red['color'].apply(cleanCol)

    # Get the clean seasons.
    df_red['season_clean'] = df_red['season'].apply(cleanSeas)

    # Separate description and fittype/bugelfrei.
    df_red['descr_clean'] = df_red['descr'].apply(delFit)
    df_red['fit'] = df_red['descr'].apply(findFit)
    df_red['bugelfrei'] = df_red['descr'].apply(findBugel)

    # Add the fittype, pattern and bugelfrei to features as well as
    # color_clean and season_clean.
    sel = df_red[['features', 'pattern', 'fit', 'bugelfrei', 'color_clean',
                  'season_clean']]
    df_red['features_tmp'] = sel.apply(extFeat, axis=1)

    # Get clean hierarchy.
    df_red['hierarchy_clean'] = df_red['hierarchy_1'] + '/' + df_red[
        'hierarchy_2'] + '/' + df_red['descr_clean'] + '/' + df_red[
        'globus_id'].apply(str)

    # Clean overall data frame.
    df_red.loc[:, df_red.columns != 'hierarchy_clean'] = \
        df_red.loc[:, df_red.columns != 'hierarchy_clean'].applymap(
            lambda x: cleanDf(x))

    # Get list of features that occur less then 10 times.
    lst_feat_rare = listRareFeat(df_red['features_tmp'].to_frame())

    # Clean the features.
    df_red['features_tmp_2'] = df_red['features_tmp'].apply(cleanFeat,
                                                            rare=lst_feat_rare)

    # Get rid of descriptions in features and vice versa.
    sel = df_red[['descr_clean', 'features_tmp_2']]
    df_red['features_clean'] = sel.apply(descFeat, axis=1)

    # Define the image path.
    df_red['img_path'] = df_red['hierarchy_clean'].apply(
        lambda x: path_img + str(Path(x).parents[1]) + '/' +
                  x.split('/')[-1] + '.png')

    # Add a column with image classification.
    df_red['img_class'] = df_red['hierarchy_clean'].apply(
        os.path.dirname).str.replace('/', '_').str.replace('-', '')

    # Get the unique colors in a separate data frame.
    colors = pd.DataFrame(data=df_red['color'].unique(), columns=[
        'colors_unique'])

    # Get reduced data frame for selected columns.
    df_red_sel = df_red[cols_sel]

    # Split it to train/validation and test data.
    train_val = df_red_sel.sample(frac=0.8, random_state=200)
    test = df_red_sel[~df_red_sel.index.isin(train_val.index)]

    return df_red, df_red_sel, train_val, test, colors


def cleanDf(x):
    '''
    Method to clean a string from 'umlaute' as well as special characters.
    '''

    # Check whether it is a string. Otherwise,
    # return without modification.
    if isinstance(x, str):
        # Clean the string.
        s = x.lower()
        s = s.replace('ä', 'a')
        s = s.replace('ö', 'o')
        s = s.replace('ü', 'u')
        s = re.sub(r'[^a-z0-9]', '', s)

        return s

    else:
        return x


def listRareFeat(df):
    '''
    Method to only keep the features that are found at least a minimum
    number of times.
    '''

    # Append all lists in the column.
    whole_lst = [i for s in df['features_tmp'].tolist() for i in s]

    # Count the number of occurence and sort according to the number of
    # times a feature occurs.
    count_sort = Counter(whole_lst).most_common()

    # Only select the ones that occur a minimum number of times.
    rare = [t[0] for t in count_sort if t[1] < 1000]

    # Clean the list of non-rare features.
    rare = cleanFeat(rare)

    return rare


def cleanSeas(x):
    '''
    Method to transform the season letter into a word.
    '''

    # W = 'winter'.
    if x == 'W':
        s = 'winter'

    # S = 'sommer'.
    elif x == 'S':
        s = 'sommer'

    # B = 'beidesaison'.
    elif x == 'B':
        s = 'beidesaison'

    else:
        s = np.nan

    return s


def cleanFeat(x, rare=None):
    '''
    Method to clean the features.
    '''

    # Initiate an empty list.
    feat_c = []

    # Loop over all the features in the list.
    for f in x:
        # Get rid ot the 'pim-', if present.
        if 'pim-' in f:
            f_c = re.sub('pim-', '', f)
        else:
            f_c = f

        # Get rid of 'umlaute' and special charactrers.
        f_c = f_c.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
        f_c = re.sub(r'[^a-zA-Z0-9]', '', f_c).lower()

        # Correctly spell 'mantel'.
        if 'mntel' in f_c:
            f_c = f_c.replace('mntel', 'mantel')

        # Append the features to the list.
        feat_c.append(f_c)

    # Make the list of features unique.
    feat_c = list(set(feat_c))

    # Remove the rare features, if provided.
    if rare is not None:
        feat_c_exr = [t for t in feat_c if t not in rare]
        return feat_c_exr
    else:
        return feat_c


def descFeat(row):
    '''
    Method to remove features that are in the description or where the
    description is in the features.
    '''

    # Initiate an empty list.
    feat_c = []

    # Loop through the features.
    for f in row['features_tmp_2']:

        # Only do this check for features that a longer than three characters.
        if len(f) > 3:
            if not row['descr_clean'] in f and not f in row['descr_clean']:
                feat_c.append(f)
        else:
            feat_c.append(f)

    return feat_c


def extFeat(row):
    '''
    Method to add content of columns to the list of features.
    '''

    # Define the list of features.
    feat = row['features']

    # Add pattern to features, if available.
    if not pd.isna(row['pattern']):
        feat.append(row['pattern'])

    # Add the fit-type to features, if available.
    if not pd.isna(row['fit']):
        feat.append(row['fit'])

    # Add 'bugelfrei' to the features, if available.
    if not pd.isna(row['bugelfrei']):
        feat.append(row['bugelfrei'])

    # Add the color to the features, if available.
    if not pd.isna(row['color_clean']):
        feat.append(row['color_clean'])

    # Add the season to the features, if available.
    if not pd.isna(row['season_clean']):
        feat.append(row['season_clean'])

    return feat


def patFromCol(x):
    '''
    Method to extract the pattern from color.
    '''

    # Check whether the color contains on of
    # the specified patterns. If not, return
    # NaNs.
    for p in pat:
       if p in x:

           return p

    return np.nan


def cleanCol(x):
    '''
    Method to remove pattern (see above) from
    the colors.
    '''

    # If a pattern is in the color, delete it.
    for p in pat:
       if p in x:
           x = x.replace(p, '')

    return x


def delFit(x):
    '''
    Method to delete the fit-type (see below) and other keywords
    from the string provided.
    '''

    # If one of the keywords is in the string provided,
    # remove the keyword.
    for s in fit_types  + ['bugelfrei', 'ausseide', '3fur2']:
       if s in x:
           x = x.replace(s, '')

    return x


def findFit(x):
    '''
    Method to find the fit-type in the string provided.
    '''

    # For every fit_type specified, collect it if in string.
    for s in fit_types:
       if s in x:

           return s

    return np.nan


def findBugel(x):
    '''
    Method to find 'bugelfrei' in the string provided.
    '''

    # If 'bugelfrei' in string, collect it.
    for s in ['bugelfrei']:
       if s in x:

           return s

    return np.nan


if __name__ == '__main__':

    # Define path to the raw csv and the images.
    path_p = './../a_data_extraction/'
    path_img = './images_small_new/'

    # Read the data.
    data = pd.read_csv(path_p + 'meta_all.csv')
    data.set_index('unique_id', inplace=True)
    data['features'] = data['features'].apply(literal_eval)

    # Clean the data.
    data_c, data_c_red, data_c_train, data_c_test, colors = \
        data_clean(data, path_img)

    # Dump data frames to CSV.
    data_c.to_csv('meta_clean.csv')
    data_c_red.to_csv('meta_clean_red.csv')
    data_c_train.to_csv('meta_clean_red_train.csv')
    data_c_test.to_csv('meta_clean_red_test.csv')
    colors.to_csv('colors_unique.csv')
