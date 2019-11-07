import pandas as pd
import os
import json
import uuid
import sys
import warnings
import requests


# Proper header to get the data.
head = ['globus_id', 'descr', 'name', 'gender', 'source_color', 'color',
        'url', 'material', 'features', 'season', 'hierarchy_full',
        'hierarchy_1', 'hierarchy_2', 'hierarchy_3', 'hierarchy_4',
        'hierarchy_5', 'hierarchy_6', 'href', 'prod_id']

# Headers used for rearranging the columns.
head_c = head.copy()
head_c.remove('material')


def parsing(path, token):

    # Get all the json files in the directory.
    json_lst = get_json_files(path)

    # Create empty data frame.
    df = pd.DataFrame(columns=head)

    # Filter the json file for the data needed and
    # append the data to a data frame.
    for j in json_lst:
        # Filter the json file.
        fil = filter_json(path + j, token)

        # Concat.
        df = pd.concat([df, fil], axis=0)

    # Rearrane the columns.
    df = df[head_c]

    # Save as csv.
    df.to_csv('test.csv')

    return


def filter_json(file_path, token):

    # Get the raw content for the json file.
    j_raw = load_json(file_path)

    # Create an empty data frame and set material, gender and
    # features to objects.
    df_tmp = pd.DataFrame(columns=head)
    df_tmp.index.name = 'unique_id'

    # Loop over all items in the json.
    for i in j_raw['items']:

        for p in i['productItemGroups']:

            # Loop over media, if existing.
            if 'media' in p.keys():

                for m in p['media']:

                    # Generate unique ID.
                    id_tmp = uuid.uuid4()

                    # Initiate the data frame.
                    df_tmp.append(pd.Series(name=id_tmp))

                    # Get the Globus ID.
                    try:
                        id_split = m['formats'][0]['url'].split('/')
                        df_tmp.loc[id_tmp, 'globus_id'] = int(id_split[-2])
                    except:
                        warnings.warn('--> NO Globus ID found.')

                    # Get source color.
                    try:
                        df_tmp.loc[id_tmp, 'source_color'] = \
                            p['manifestation']['color']['sourceColor'][
                                'label']['de']
                    except:
                        warnings.warn('--> NO source color found.')

                    # Get the color.
                    try:
                        df_tmp.loc[id_tmp, 'color'] = \
                            p['manifestation']['color']['label']['de']
                    except:
                        warnings.warn('--> NO color found.')

                    # Get the url.
                    try:
                        df_tmp.loc[id_tmp, 'url'] = m['formats'][0]['url']
                    except:
                        warnings.warn('--> NO url found.')

                    # Get the material features.
                    try:
                        df_tmp.loc[id_tmp, 'material'] = [ma['enum'] for ma in
                                                          i['material'][
                                                              0]['values']]
                    except:
                        warnings.warn('--> NO material found.')

                    # Get the href.
                    try:
                        df_tmp.loc[id_tmp, 'href'] = i['href']
                    except:
                        warnings.warn('--> NO href found.')

                    # Get the product item group id.
                    try:
                        df_tmp.loc[id_tmp, 'prod_id'] = p['productItemGroupId']
                    except:
                        warnings.warn('--> NO product ID found.')

                    # Get the features, gender and the description.
                    feat_tmp = []
                    try:
                        for f in i['features']:
                            if 'produkttyp' in f['key']:
                                try:
                                    df_tmp.loc[id_tmp, 'descr'] = f['values'][0][
                                        'enum']
                                except:
                                    pass
                                try:
                                    df_tmp.loc[id_tmp, 'descr'] = f['values'][
                                        'label']['de']
                                except:
                                    warnings.warn('--> NO description found.')

                            if 'name' in f['key']:
                                try:
                                    df_tmp.loc[id_tmp, 'name'] = f['values'][0][
                                        'label']['de']
                                except:
                                    warnings.warn('--> NO name found.')

                            if 'produkttyp' not in f['key'] and 'gender' not in f[
                                'key'] and 'name' not in f['key']:
                                try:
                                    if isinstance(int(f['values'][0]['enum']),
                                                  int):
                                        warnings.warn('--> NO feature: {}'.format(
                                            f['key']))
                                except:
                                    if 'enum' in f['values'][0].keys():
                                        feat_raw = [ma['enum'] for ma in f[
                                            'values']][0]
                                        # Only use features with length > 2.
                                        if len(feat_raw) > 2.0:
                                            feat_tmp.append(feat_raw)
                                    else:
                                        pass

                        # Insert the features.
                        df_tmp.loc[id_tmp, 'features'] = feat_tmp

                    except:
                        warnings.warn('--> NO features found.')

                    # Get the hierarchy.
                    try:
                        h_tmp = i['categories'][0]['key']
                        h_tmp_lst = h_tmp.split('.')

                        df_tmp.loc[id_tmp, 'hierarchy_full'] = h_tmp

                        for n in range(2, len(h_tmp_lst) + 1):
                            c = 'hierarchy_' + str(n - 1)

                            df_tmp.loc[id_tmp, c] = h_tmp_lst[n]
                    except:
                        pass

                    # Get the gender.
                    try:
                        if 'damen' in i['categories'][0]['key']:
                            df_tmp.loc[id_tmp, 'gender'] = 'damen'
                        elif 'herren' in i['categories'][0]['key']:
                            df_tmp.loc[id_tmp, 'gender'] = 'herren'
                        elif 'kinder' in i['categories'][0]['key']:
                            df_tmp.loc[id_tmp, 'gender'] = 'kinder'
                    except:
                        warnings.warn('--> NO gender found.')

                    # Get the season.
                    try:
                        df_tmp.loc[id_tmp, 'season'] = i['SAP']['saison']
                    except:
                        warnings.warn('--> NO season found.')

            else:

                # Generate unique ID.
                id_tmp = uuid.uuid4()

                # Initiate the data frame.
                df_tmp.append(pd.Series(name=id_tmp))

                # Get the Globus ID.
                try:
                    id_split = p['media'][0]['formats'][0]['url'].split('/')
                    df_tmp.loc[id_tmp, 'globus_id'] = int(id_split[-2])
                except:
                    warnings.warn('--> NO Globus ID found.')

                # Get source color.
                try:
                    df_tmp.loc[id_tmp, 'source_color'] = \
                        p['manifestation']['color']['sourceColor'][
                            'label']['de']
                except:
                    warnings.warn('--> NO source color found.')

                # Get the color.
                try:
                    df_tmp.loc[id_tmp, 'color'] = \
                        p['manifestation']['color']['label']['de']
                except:
                    warnings.warn('--> NO color found.')

                # Get the url.
                try:
                    df_tmp.loc[id_tmp, 'url'] = p['media'][0]['formats'][
                        0]['url']
                except:
                    warnings.warn('--> NO url found.')

                # Get the material features.
                try:
                    df_tmp.loc[id_tmp, 'material'] = [ma['enum'] for ma in
                                                      i['material'][
                                                          0]['values']]
                except:
                    warnings.warn('--> NO material found.')

                # Get the href.
                try:
                    df_tmp.loc[id_tmp, 'href'] = i['href']
                except:
                    warnings.warn('--> NO href found.')

                # Get the product item group id.
                try:
                    df_tmp.loc[id_tmp, 'prod_id'] = p['productItemGroupId']
                except:
                    warnings.warn('--> NO product ID found.')

                # Get the features, gender and the description.
                feat_tmp = []
                try:
                    for f in i['features']:
                        if 'produkttyp' in f['key']:
                            try:
                                df_tmp.loc[id_tmp, 'descr'] = f['values'][0][
                                    'enum']
                            except:
                                pass
                            try:
                                df_tmp.loc[id_tmp, 'descr'] = f['values'][
                                    'label']['de']
                            except:
                                warnings.warn('--> NO description found.')

                        if 'name' in f['key']:
                            try:
                                df_tmp.loc[id_tmp, 'name'] = f['values'][0][
                                    'label']['de']
                            except:
                                warnings.warn('--> NO name found.')

                        if 'produkttyp' not in f['key'] and 'gender' not in f[
                            'key'] and 'name' not in f['key']:
                            try:
                                if isinstance(int(f['values'][0]['enum']),
                                              int):
                                    warnings.warn('--> NO feature: {}'.format(
                                        f['key']))
                            except:
                                if 'enum' in f['values'][0].keys():
                                    feat_raw = [ma['enum'] for ma in f[
                                        'values']][0]
                                    # Only use features with length > 2.
                                    if len(feat_raw) > 2.0:
                                        feat_tmp.append(feat_raw)
                                else:
                                    pass

                    # Insert the features.
                    df_tmp.loc[id_tmp, 'features'] = feat_tmp

                except:
                    warnings.warn('--> NO features found.')

                # Get the hierarchy.
                try:
                    h_tmp = i['categories'][0]['key']
                    h_tmp_lst = h_tmp.split('.')

                    df_tmp.loc[id_tmp, 'hierarchy_full'] = h_tmp

                    for n in range(2, len(h_tmp_lst) + 1):
                        c = 'hierarchy_' + str(n - 1)

                        df_tmp.loc[id_tmp, c] = h_tmp_lst[n]
                except:
                    warnings.warn('--> NO hierarchy found.')

                # Get the gender.
                try:
                    if 'damen' in i['categories'][0]['key']:
                        df_tmp.loc[id_tmp, 'gender'] = 'damen'
                    elif 'herren' in i['categories'][0]['key']:
                        df_tmp.loc[id_tmp, 'gender'] = 'herren'
                    elif 'kinder' in i['categories'][0]['key']:
                        df_tmp.loc[id_tmp, 'gender'] = 'kinder'
                except:
                    warnings.warn('--> NO gender found.')

                # Get the season.
                try:
                    df_tmp.loc[id_tmp, 'season'] = i['SAP']['saison']
                except:
                    warnings.warn('--> NO season found.')

    # Add materials to features.
    df_tmp['features'] = df_tmp['features'] + df_tmp['material']
    df_tmp.drop(['material'], axis=1, inplace=True)

    # Drop all rows that do not have an url and do not have
    # a feature.
    df_tmp.dropna(subset=['url'], inplace=True)
    df_tmp.dropna(subset=['features'], inplace=True)

    # Fill NaNs of 'descr' with 'others'.
    df_tmp['descr'].fillna('others', inplace=True)

    # Delete duplicates in features.
    df_tmp['features'] = df_tmp.apply(lambda row: list(set(row['features'])),
                                      axis=1)

    # Download and save the pictures.
    download_pictures(df_tmp, token)

    # Save to csv.
    df_tmp.to_csv(json_path.replace('.json', '.csv'))

    return df_tmp


def download_pictures(df, token):

    # Loop over all the rows.
    for i, r in df.iterrows():

        # Create directory.
        try:
            dir_tmp = 'pictures/'+r['hierarchy_1']+'/'+r['hierarchy_2']+'/'
        except:
            try:
                dir_tmp = 'pictures/'+r['hierarchy_1']+'/'
            except:
                dir_tmp = 'pictures/'

        # Get saving name of image.
        i_name = str(r['globus_id']) + '.webp'

        # Check whether the directory exist. If not, create the directory.
        if not os.path.isdir(dir_tmp):
            os.makedirs(dir_tmp)

        # Define the head for accessing the API.
        head = {'Authorization': 'Bearer ' + token}

        # Download and save the picture.
        file = requests.get(r['url'], headers=head)
        open(dir_tmp + i_name, 'wb').write(file.content)

    return


def load_json(file_path):

    # Read the json.
    with open(file_path) as f:
        j_tmp = json.load(f)

    return j_tmp


def get_json_files(path):

    # List of all json files in the directory
    lst = [p for p in os.listdir(path) if
           p.endswith('.json')]

    return lst


if __name__ == '__main__':

    # # Define the json path.
    # json_path = './test.json'
    #
    # # Access to API.
    # myToken = str(pd.read_csv('token_file.txt',
    #                           header=None).values.squeeze())
    #
    # filter_json(json_path, myToken)

    json_path = sys.argv[1]
    token = sys.argv[2]

    filter_json(json_path, token)