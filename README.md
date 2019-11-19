# globus_project
Propulsion project to classify images for Globus that should be deployed afterwards.

### General workflow:

- ***a_data_parsing:*** Download jsons with picture details using Globus API

- ***b_data_parsing:*** Get image description and image link from jsons, download images

- ***c_data_cleaning:*** Clean image description, form two subgroups for training: categories and features. Categories are based on hierarchies of Globus (e.g. highest is geneder, second is general fashion item, third more detailed). Features include color, material, season, pattern and miscellaneous

- ***d_model:*** Build branched convolutional neural network to predict categories and features simultaneousely

- ***e_web:*** Flask based user interface to check model predictions
