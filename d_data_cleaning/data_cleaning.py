import pandas as pd
import numpy as np
import re
from ast import literal_eval
from collections import Counter


# Define list used for the data cleaning.
cat = ['taschen', 'schuhe', 'pullover-strick', 'top-shirts-sweats',
       'schals', 'hemden', 'hosen', 'oberteile', 'blusen-tuniken',
       'shirts-tops-sweats', 'kleider', 'jacken', 'tagwasche',
       'strumpfmode', 'loungewearyoga', 'unterteile', 'wasche-pyjamas',
       'wasche', 'bademode', 'socken', 'jeans', 'mantel',
       'krawatten-fliegen-pochetten', 'kopfbedeckungen', 'kleider-sets',
       'nachtwasche', 'jupe', 'blazer', 'vestons', 'shorts-bermudas',
       'gurtel', 'handschuhe', 'westen-gilets', 'anzuge', 'poncho-cape',
       'lederjacken-mantel', 'overalls', 'morgen-bademantel',
       'lederjacken-ledermantel', 'shapewear']

fit_types = ['slimfit','regularfit','skinnyfit','loosfit','taperedfit']

pat = ['gestreift', 'klein gemustert', 'kariert', 'karo', 'Glattleder',
       'Lack']

cols_sel = ['globus_id', 'descr_clean', 'hierarchy_clean',
            'gender', 'source_color', 'color_clean', 'season',
            'features_clean']


def data_clean(df):

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

    # Add the fittype, pattern and bugelfrei to features.
    sel = df_red[['features', 'pattern', 'fit', 'bugelfrei', 'color_clean',
                  'season_clean']]
    df_red['features_tmp'] = sel.apply(extFeat, axis=1)

    # Clean overall data frame.
    df_red = df_red.applymap(lambda s: s.lower() if type(s) == str else s)
    df_red = df_red.applymap(lambda s: s.replace('ä', 'a') if type(s) == str
                             else s)
    df_red = df_red.applymap(lambda s: s.replace('ö', 'o') if type(s) == str
                             else s)
    df_red = df_red.applymap(lambda s: s.replace('ü', 'u') if type(s) == str
                             else s)
    df_red = df_red.applymap(lambda s: re.sub(r'[^a-z0-9]', '', s)
        if type(s) == str else s)

    # Get clean hierarchy.
    df_red['hierarchy_clean'] = df_red['hierarchy_1'] + '/' + df_red[
        'hierarchy_2'] + '/' + df_red['descr_clean'] + '/' + df_red[
        'globus_id'].apply(str)

    # Get list of features that occur less then 10 times.
    lst_feat_rare = listRareFeat(df_red['features_tmp'].to_frame())

    # Clean the features.
    df_red['features_tmp_2'] = df_red['features_tmp'].apply(cleanFeat,
                                                            rare=lst_feat_rare)

    # Get rid of descriptions in features and vice versa.
    sel = df_red[['descr_clean', 'features_tmp_2']]
    df_red['features_clean'] = sel.apply(descFeat, axis=1)

    # Get reduced data frame for selected columns.
    df_red_sel = df_red[cols_sel]

    # Split it to train/validation and test data.
    train_val = df_red_sel.sample(frac=0.8, random_state=200)
    test = df_red_sel[~df_red_sel.index.isin(train_val.index)]

    return df_red, df_red_sel, train_val, test


def listRareFeat(df):

    whole_lst = [i for s in df['features_tmp'].tolist() for i in s]

    count_sort = Counter(whole_lst).most_common()

    rare = [t[0] for t in count_sort if t[1] < 10]

    rare_cl = cleanFeat(rare)

    return rare_cl


def cleanSeas(x):

    if x == 'W':
        s = 'winter'

    elif x == 'S':
        s = 'sommer'

    elif x == 'B':
        s = 'beidesaison'

    else:
        s = np.nan

    return s


def cleanFeat(x, rare=None):

    feat_c = []

    for f in x:
        if 'pim-' in f:
            f_c = re.sub('pim-', '', f)
        else:
            f_c = f

        f_c = f_c.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
        f_c = re.sub(r'[^a-zA-Z0-9]', '', f_c).lower()

        if 'mntel' in f_c:
            f_c = f_c.replace('mntel', 'mantel')

        feat_c.append(f_c)

    feat_c = list(set(feat_c))

    if rare is not None:
        feat_c_exr = [t for t in feat_c if t not in rare]
        return feat_c_exr
    else:
        return feat_c


def descFeat(row):

    feat_c = []

    for f in row['features_tmp_2']:

        if len(f) > 3:
            if not row['descr_clean'] in f and not f in row['descr_clean']:
                feat_c.append(f)
        else:
            feat_c.append(f)

    return feat_c


def extFeat(row):

    feat = row['features']

    if not pd.isna(row['pattern']):
        feat.append(row['pattern'])

    if not pd.isna(row['fit']):
        feat.append(row['fit'])

    if not pd.isna(row['bugelfrei']):
        feat.append(row['bugelfrei'])

    if not pd.isna(row['color_clean']):
        feat.append(row['color_clean'])

    if not pd.isna(row['season_clean']):
        feat.append(row['season_clean'])

    return feat


def patFromCol(x):

    for p in pat:
       if p in x:
           return p
    return np.nan


def cleanCol(x):

    for p in pat:
       if p in x:
           x = x.replace(p, '')
    return x


def delFit(x):

    for s in fit_types  + ['bugelfrei', 'ausseide', '3fur2']:
       if s in x:
           x = x.replace(s, '')
    return x


def findFit(x):

    for s in fit_types:
       if s in x:
           return s
    return np.nan


def findBugel(x):

    for s in ['bugelfrei']:
       if s in x:
           return s
    return np.nan


if __name__ == '__main__':

    # Define path to the raw csv and path to the saving location.
    path_p = '/home/fabiolutz/propulsion/globus_project/c_data_parsing/'
    path_s = '/home/fabiolutz/propulsion/globus_project/d_data_cleaning/'

    # Read the data.
    data = pd.read_csv(path_p + 'meta_all.csv')
    data.set_index('unique_id', inplace=True)
    data['features'] = data['features'].apply(literal_eval)

    # Clean the data.
    data_c, data_c_red, data_c_train, data_c_test = data_clean(data)

    # Save it to another csv.
    data_c.to_csv(path_s + 'meta_clean.csv')
    data_c_red.to_csv(path_s + 'meta_clean_red.csv')
    data_c_train.to_csv(path_s + 'meta_clean_red_train.csv')
    data_c_test.to_csv(path_s + 'meta_clean_red_test.csv')