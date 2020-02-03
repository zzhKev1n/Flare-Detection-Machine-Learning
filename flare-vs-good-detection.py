import cv2
import tensorflow as tf
import os
import numpy as np
from termcolor import colored
from tensorflow.keras.models import model_from_json

overall_category = ["flare", "good"]

# define a function to load the path of the files
# and then process the image

file_path = input(colored('Please type the full path of the folder which stored the image files for tests: \n', 'red'))

# check if the path is valid
# if the input path is valid, then proceeed to the image data loading process
# if the input path is not valid, then ask again
path_validation = False
while path_validation != True:
	try:
		path_validation = os.path.exists(file_path)
		if path_validation == False:
			file_path = input(colored('The file path does not exist, please try again. \n', 'red'))
	except Exception:
		file_path = input(colored('The file path does not exist, please try again. \n', 'red'))

# load the images from the folder and process the data
loaded_resized_data = []
num_of_img = 0
img_name = []
for img in os.listdir(file_path):
	try:
		img_size = 70
		img_array = cv2.imread(os.path.join(file_path, img), cv2.IMREAD_GRAYSCALE)
		resized_img_array = cv2.resize(img_array, (img_size, img_size))
		loaded_resized_data.append([resized_img_array])
		img_name.append(img)
	except Exception:
		pass 	# pass the non-image files 

# reshape the image data
X_test = np.array(loaded_resized_data).reshape(-1, img_size, img_size, 1)

# load the trained model from json file
file_path = os.path.dirname(os.path.realpath(__file__))
model = tf.keras.models.load_model(os.path.join(file_path, "training/CNN.model"))
# file_path = os.path.dirname(os.path.realpath(__file__))
# json_file = open(os.path.join(file_path, "training/model.json"), 'r')
# loaded_json_file = json_file.read()
# json_file.close()
# model = model_from_json(loaded_json_file)

# predict the loaded images 
prediction = model.predict([X_test])

# print out the results
for l in range(len(img_name)):
	result = "The classification of {} is {}".format(img_name[l], overall_category[int(prediction[l][0])])
	print(result)