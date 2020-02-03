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

# set up the gpu options
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

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

# test the layers in different conbinations to find the optimal one
conv_layers = [1, 2, 3]
layer_sizes = [32, 64, 128]
dense_layers = [0, 1, 2]

# test the different combinations between the layers in the loop
for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:

			# label the each combination
			name = "{}-conv-{}-nodes-{}-dense-{}time".format(conv_layer, layer_size, dense_layer, int(time.time()))
			tensorboard = TensorBoard(log_dir=os.path.join(file_path, 'logs/{}'.format(name)))
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