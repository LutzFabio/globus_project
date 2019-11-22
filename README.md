############################################################################################
############################################################################################
# Enhancing Fashion Business Processes using AI for [Globus](https://www.globus.ch)
############################################################################################
############################################################################################

This is the official documentation for the final project we did for Globus after the Data
Science bootcamp at Propulsion Academy in Zurich.

## Problem description
Globus is a fashion retailer that besides its stores (such as the flagship store at Zurich
Bahnhofstrasse) also sells fashion products through their online store (www.globus.ch).
The online store holds thousands of images representing the products together with description.
Currently, these pictures are classified (t-shirt, shoe, etc.) and described (color, texture, etc.)
manually by employees, that takes many of their valuable working hours for such rather
monotonuous work.

## Goal
Globus approached Propulsion Academy to explore whether the daily life of their employees 
could be made more meaningful. More precisely, Globus wanted us to develop AI prototype 
algorithms that could do part of the monotonuous work for the employees such that more 
of the working time could be devoted to more interessting and more valuable tasks.

## Summary of results
A prototype has been successfully developed by using deep learning techniques (Convolutional Neural Networks).
The results show clearly that having AI algorithms supporting Globus in classifying and describing 
fashion images is possible. Since there was only less than four weeks of time (incl. data querying and data cleaning), 
the algorithm is not yet fully developped and can be further improved. Ideas how to do it, will be
presented later in this document.

## Workflow description

### main folder
In the main folder, besides this README file, a licence can be found and the environment
("environment.yml") which can be installed using "conda" and includes all the packages 
that are needed to execute the code.

### a_data_parsing
In this folder, all the code snippets can be found that were needed to extract the raw data
from the API of Globus. The steps included were the following:

	1. The raw JSON files had to be downloaded in which the information to all the
	   the pictures are stored and the links to the pictures were given. This was done by
	   using a bash script together with a token file (a token has to be generated and is
	   then valid for 24 hours). 
	   --> Download_jsons.sh and token_file.txt

	2. The downloaded JSON files are afterwards being parsed, i.e. the information for
	   each of the images is read and stored and csv file (one for every JSON file). At
	   the same time, the weblink is used for downloading the pictures. All this is being
	   done by a bash script that calls a python script, which eventually carries out the
	   necessary tasks. For downloading the pictures, a token is needed as well to access
	   the API (again valid for 24 hours).
	   --> Download_images.sh, json_parsing.py and token_file_2.txt

	3. The CSV's are then put together to one large CSV file. This is done again using
	   a bash script.
	   --> merge_csv.sh 

	4. As a last step, the images are moved to a specified folder structure, are resized
	   to identical sizes and are converted to PNG format (needed for later processing,
	   because WEBP format is not supported by machine learning libraries). This is again
	   done by a bash script.
	   --> Resize_move_images.sh

### b_data_cleaning
In this folder, all the code is found that is used for cleaning the raw data downloaded in
the previous step. By accessing the CSV file that contains all the image information (see
previous section), a clean data set is being generated. Clean means for example that a
the items to be included can be selecte, a clean hierarchy (e.g. damen/kleider/kleid) is
generated, patterns were cut from color and assigned to features, everything was converted
to lower case, and so on. Most importantly, a clean features and a clean hierarchy are
needed for the alorithm in the end. The end format is a Pandas data frame that contains a
column "img_class" that contains the clean hierarchies and a column "features_clean" that
contains in each row a list of strings with the features. All that is handled by a single
Python script ("data_cleaning.py").

There is another Python script found in the folder ("checkimagesFORtransparency.py") which
checks images whether some transparency is contained. The main thinking behind this script
is that images that only show a fashion item from very close (in order to see the texture)
should be excluded since these may not be associated with a particular category. This
script, however, is currently not implemented in the main process but could be done in the
future, if needed.

### c_model
This folder contains the code that trains Convolutional Neural Networks based on the clean
data generated in the previous step. In order to to both, categorizing fashion images (e.g. t-shirt, jacket, etc.) and describing their features (e.g. color, texture, etc.) is a 
different task but nevertheless is somehow related we decided to implement a single-input,
multi-output structure. The main or best model applies transfer learning using the ResNet50
model and adding some own layers. The structure is the following:

				input
				  |
				  |
			       ResNet50
				  |
				  |
			  GlobalAveragPooling	
				 / \	
				/   \
			       /     \
			      /	      \
		             /         \
			 Dense         Dense
			   |             |
	                   |             |
			Dropout       Dropout
			   |	         |
	                   |             |
	                 Output        Output
		      (categories)   (features)
  
This model by retraining the last two blocks of the ResNet50 model (~layer 143 to the end) with a learning rate of 0.001, the Adam optimizer and "Relu" activation for the intermediate
Dense layer produced good results. 

Besides the above specified structure, also an self-constructure Convolutional Neural
Network structure exists. This can be trained by simply changing "model_type" in the
settings from "r50" (ResNet50) to "own". 

Also to mention is that most of training of a model can be easily steered by simply
making changes in the setting section at the top of the script "cnn_globus.py". 

In terms of output, the model is saved after every fifth epoch (as currently specified 
like that in the settings), is validated after every epoch (as currently spedified like
that in the settings) and the losses and accuracies are saved in a log file. Furthermore,
the model is saved at the end together with various self-created outputs, such as 
confusion matrices, validation predictions, etc.

### d_web
This folder contains all the necessary code snippets for running the UI created for this
project. The UI allows to upload images and the predict them based on several underlying
models and provides a visual probability tree for the classification of the images. Further,
the UI also shows the features that can be added or removed upon description. In order to start the UI, navigate to this folder in the command line and type the following command: 
"FLASK_ENV=development flask run". After running this command, the address is displayed
in the command window that can be opened in the browser.

When running the UI, it can happen that an error occures that complains about "tensorflow",
"keras" or a combination thereof. In order to solve this problem, tensorflow has to be
installed on the computer itself (and not only in the conda environment). 

## Challenges faced
During the process of implementing this prototype, we faced the follwing challenges that$
are noteworthy to share:

	1. The size of the data is too big to load into memory. For that we were working
	   with the ImageDataGenerator from Keras that provides functionality to load
	   images in batches.

	2. However, the existing libraries are mostly made for simple model architecture.
	   As we found out, the ImageDataGenerator is not made for multi-output models, 
	   only for single-output models. So the combination of a large size of data
	   together with a complex architecture proved especially difficult. We solved
	   this by implementing a custom data generator.

## Suggestions for improving the model
Since there were only less than four weeks of time, the model is not fully developed so
far and we had many more ideas how to further improve it. The most important ones are the
following:

	1. Split the features into groups (colors, texture, material, etc.) and create a
	   separate output branch for each feature.

	2. Apply a custom loss function and accuracy score for features. The currently
	   used "binary_crossentropy" with "binary_accuracy" is by default very low and
	   very high, respectively. This because there are many features but only a few of
	   them are present for a particular product. That means that a correct predction
	   of all the zeros is also counted as accuracy, thus leading to a high accuracy
	   and low loss.

	3. Another idea for a custom loss function an accuracy score would be to include
	   not only the lowest level of the hierarchy but also the predictions on the
	   upper level (e.g. herren, damen, etc.).

	4. Pictures the are taken from very close and only show the material could be
	   removed. In order to do so, a code snipped is provided for that such that
	   this could be implemented in the running infrastructure.

	5. For time reasons, we were not able to test a lot of layer structures. Maybe
	   the performance can be improved by adding more layers, using a different
	   dropout rate, etc.

	6. The underlying data could be made more balanced by intentionally increasing
	   the amount of images for categories / features less occuring. This because
	   making the data balanced on the feature side cannot be obtained by simply
	   up- or downsampling (since the same feature is present for several items).

	7. Even though we have invested a lot of time to clean the features, the features
	   could still be cleaned further.

## Final word
At this point, we would like to thank Globus for giving us such a challenging and
interesting project to us. We learned a whole lot during these weeks and hope that our
prototype builds a good base for them in order to make their internal processes more
efficient.

Masha & Fabio

