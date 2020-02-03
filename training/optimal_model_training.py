# this script is the final machine training with the optimal combination between different layers
# the optimal combination is found with the locate_optimal_layers python script
import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import os
import numpy as np
import time

# disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# locate the file path for pickle loading and TensorBoard output
file_path = os.path.dirname(os.path.realpath(__file__))

# load stored X data and process the data into tensor
# scale down the X value for the image training
# the maximum image pixal value is 255
pickle_in = open(os.path.join(file_path,"X.pickle"), "rb")
X = pickle.load(pickle_in)
X = X/255.0
X = tf.convert_to_tensor(X)

# load and process Y value
pickle_in = open(os.path.join(file_path,"Y.pickle"), "rb")
Y = pickle.load(pickle_in)
Y = np.array(Y)

# the optimal combination is 1 conv layers and 2 dense layer with the layer size of 128
conv_layers = [1]
layer_sizes = [128]
dense_layers = [2]

# create layers
for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:

			# plot the machine learning progress of each epoch using TensorBoard
			name = "flare-vs-good-{}".format(int(time.time()))
			tensorboard = TensorBoard(log_dir=os.path.join(file_path, 'optimal2-log/{}'.format(name)))
			print(name)

			model = Sequential()

			# input the image data with conv layer
			model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))

			# adding the conv layer if it is more than 1
			for l in range(conv_layer-1):
				model.add(Conv2D(layer_size,(3, 3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))

			# flatten the layers for the application of dense layers
			model.add(Flatten())

			# add dense layers
			for l in range(dense_layer):
				model.add(Dense(layer_size))
				model.add(Activation('relu'))

			model.add(Dense(1))
			model.add(Activation('sigmoid'))

			model.compile(loss='binary_crossentropy',
						  optimizer='adam',
						  metrics=['accuracy'])

			model.fit(X, Y, batch_size=10, epochs=20, validation_split=0.1, callbacks=[tensorboard])

# save model
model.save('/Users/Kev1n/Python/Flare-Detection-Machine-Learning/flare_detection_CNN.model')
