import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle 

# define the direction of the file
# define the categories
data_dir = "/Users/Kev1n/Python/Flare-Detection-Machine-Learning/training"
overall_categories = ["flare", "good"]

# load the image data and resize it
training_data = []
img_size = 70
for category in overall_categories:
    path = os.path.join(data_dir, category)
    class_num = overall_categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (img_size, img_size))
            training_data.append([new_array, class_num])
        except Exception:
            pass

# in case the machine find out the data pattern in the training, shuffle the data
import random
random.shuffle(training_data)

# separate the image data and the labels
X = []
Y = []
for features, label in training_data:
    X.append(features)
    Y.append(label)

# reshape image data
X = np.array(X).reshape(-1, img_size, img_size, 1)

# save the X, Y data with pickle
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()