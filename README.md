# FERVT_CA
This is a code implementation of ZIYU ZHENG's Face Emotion Recognition thesis.We propose a new facial emotion recognition model base on FERVT model. In our model, We use coordinate attention to replace the high-level feature extraction part in the original model, aiming for a lighter architecture and enhanced capture of spatial information. Additionally, we modify the fusion of feature maps to obtain more detailed information and proposed a new loss function to help the model training.

## Installation and Usage

To run this project, you first need to make sure that your python version is greater than or equal to 10.0.0, and that your pytorch version is greater than or equal to 2.0.First you need to create two folders called checkpoints and datasests.These two folders hold the trained model and the dataset respectively.For the dataset, you can find it at www.kaggle.com. For the datasets, you need to divide them into three subfolders named train,val,test.In each subfolder it needs to be divided into 7 smaller folders to load images of different emotions.The framework is as follows

```
/dataset
│
├── train
│ ├── anger
│ ├── disgust
│ ├── fear
│ ├── happiness
│ ├── sadness
│ ├── surprise
│ └── neutral
├── val
├── test
```
## Train and Test

When everything is ready, run it in the corresponding "train.py" folder, then the corresponding confusion matrix and weight files will be generated in the checkpoints, and then test it in the "test.py" file.
