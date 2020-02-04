This project aims to classify the images from different categories with python machine learning. Its goal is to distinguish whether the photo is a flare photo or a good picture.

The program is trained under the tensorflow keras package and the training sample from each stream is evenly distributed for the optimal training result. The sample images and the model training scripts are stored under the training folder.

The data of the images from both category is loaded and processed with the 'data_loading.py' script under the 'training' folder and then the image data with its labels are saved separately as X.pickle (processed image data for training) and Y.pickle (category labels for image data). The processed data is saved as separate files so that the image data does not need to process again in the further training process.

Then the script named 'locate_optimal_layers.py' under the 'training' folder loads the saved data and then train the model with different combinations between convolutional layers, dense layers and size of the layers. the results are recorded with TensorBoard and 27 combinations are tested:

    conv_layers = [1, 2, 3]
    layer_sizes = [32, 64, 128]
    dense_layers = [0, 1, 2]    
    
The results on Tensorboard shows that the optimal combination is:

    conv_layers = [1]
    layer_sizes = [128]
    dense_layers = [2]

Therefore, the model training with the optimal combination is executed under the 'optimal_model_training.py' script under the 'training' folder. The model will be saved after the training of this script.

Finally, the classifier script named 'detector.py' is able to load the trained model and classify whether the image is flare or good. If you want to use this classifier, please put all of testing photos in one folder. Then run the classifier script and then it will require you to input the full directory to the folder where contains all the testing images. Finally, it will return the results on your screen.
