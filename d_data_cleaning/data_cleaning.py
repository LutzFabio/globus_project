import pandas as pd
import numpy as np
import re
from ast import literal_eval


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
            'gender', 'source_color', 'color_clean', 'features_clean']


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

    # Separate description and fittype/bugelfrei.
    df_red['descr_clean'] = df_red['descr'].apply(delFit)
    df_red['fit'] = df_red['descr'].apply(findFit)
    df_red['bugelfrei'] = df_red['descr'].apply(findBugel)

    # Add the fittype, pattern and bugelfrei to features.
    sel = df_red[['features', 'pattern', 'fit', 'bugelfrei']]
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

    # Clean the features.
    df_red['features_tmp'] = df_red['features_tmp'].apply(cleanFeat)

    # Get rid of descriptions in features and vice versa.
    sel = df_red[['descr_clean', 'features_tmp']]
    df_red['features_clean'] = sel.apply(descFeat, axis=1)

    return df_red, df_red[cols_sel]


def cleanFeat(x):

    feat_c = []

    for f in x:
        if 'pim-' in f:
            f_c = re.sub('pim-', '', f)
        else:
            f_c = f

        f_c = f_c.replace('ä', 'a').replace('ö', 'o').replace('ü', 'u')
        f_c = re.sub(r'[^a-zA-Z0-9]', '', f_c).lower()

        feat_c.append(f_c)

    feat_c = list(set(feat_c))

    return feat_c


def descFeat(row):

    feat_c = []

    for f in row['features_tmp']:

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
    data_c, data_c_red = data_clean(data)

    # Save it to another csv.
    data_c.to_csv(path_s + 'meta_clean.csv')
    data_c_red.to_csv(path_s + 'meta_clean_red.csv')