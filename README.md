# MachineLearningMRI
Predicts brain tumor type using CNN model. 

Set-up: The runnable code is within brainScanPrediction.ipynb. To run the program without conflicts, an older version of python has to be installed (3.11.9). The following dependencies will also need to be installed: scikit-learn, numpy, torch, torchvision, matplotlib, pandas, tensorflow, and keras_preprocessing.

Additionally, user will have to set the appropriate directory for the training and testing dataset (images). The code contains two variables called ‘train_data_dir’ and ‘test_data_dir’ that should be set to the correct path where the training and the testing dataset images can be found. These training and testing images are inside the TestingImages folder.
