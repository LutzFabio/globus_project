import pandas as pd
import numpy as np
import os
import json
from a_configs.api import *


json_path = '/home/fabiolutz/propulsion/globus_project/c_data_parsing' \
            '/raw_json_AllProducts/'


def to_db():

    # Get list of json files.
    json_lst = os.listdir(json_path)

    # Loop through the json's and store them.
    for j in json_lst:

        with open(json_path + j) as json_tmp:
            data_tmp = json.load(json_tmp)

        # Define the collections.
        coll = db['raw']

        # Save to the collections.
        coll.insert_many(data_tmp)

    return


def from_db():



    return 1


if __name__ == '__main__':

    to_db()