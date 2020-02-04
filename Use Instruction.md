1. Read the ‘package-setup-guide.md’ to setup the package and libraries

2. clone the repositories to your computer

3. Run the ‘optimal_model_training.py’ with python 3 (this file is under the ‘training’ folder). The model will be trained and stored under the ‘training’ folder

4. Place all the testing image in one folder

5. Run the ‘detector.py’ file with python 3 under the cloned folder. Then a text saying will be printed on the screen saying: 

        "Please type the full path of the folder which stored the image files for tests" 

6. Input the directory of the folder which contains all the testing images (for example: if the folder is under Download data tree and named ‘testing’ in Ubuntu 16

7. Then the testing images will be detected, and the result will printed on the screen in the following format:

        The classification of 1.jpg is flare
        The classification of 2.jpg is good
        The classification of 3.jpg is flare
        The classification of 4.jpg is good
        The classification of 5.jpg is good