# Diabetic-Retinopathy-Detection (Summer Course homework 3)

This project is to classify people's degree of diabetes through pictures of retinopathy, which is related to [kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection#description). All the models in this project were built by pytorch.

In addition, please refer to the following report link for detailed report and description of the experimental results.
https://github.com/secondlevel/Diabetic-Retinopathy-Detection/blob/main/Experiment%20Report.pdf

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/150523554-3b088451-deab-4cbc-95d8-59e5a3f40c20.png" title="training curve" hspace="350"/>
</p>

## Hardware

In this project, a total of two different CPUs and GPUs are used for model training. 

The first GPU used is NVIDIA GeForce GTX TITAN X, and the second is NVIDIA GeForce RTX 2080 Ti.  

|                 | Operating System | CPU                                      | GPU                        |
|-----------------|------------------|------------------------------------------|----------------------------|
| First machine   | Windows 10       | Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  | NVIDIA GeForce GTX TITAN X |
| Second machine  | Windows 10       | Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz | NVIDIA GeForce RTX 2080 Ti |
  
## Requirement

In this work, I use Anaconda and Pip to manage my environment.

```bash=
$ conda create --name retinopathyenv python=3.8
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install numpy
$ conda install matplotlib -y
$ conda install scikit-learn -y
$ conda install pandas -y
$ pip install torchsummary
$ pip install pkbar
$ pip install tqdm
```

## Model Architecture

In this project, I used the architecture of [**ResNet 18**](https://arxiv.org/pdf/1512.03385.pdf) and [**ResNet 50**](https://arxiv.org/pdf/1512.03385.pdf) to classify images.  

Due to the skip connection and shortcut connection methods in the ResNet architecture, gradient vanishing and exploding gradient problems are less likely to occur, and a model has lower parameters to train.

- ### skip connection

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/150625583-8db03440-2137-45fb-ae66-66c8b792a5fd.png" title="skip connection" width="400" hspace="300"/>
</p>

- ### shortcut connection

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/150625573-56d1be38-9359-4307-b8fc-9401afcd7900.png" title="shortcut connection" width="800" hspace="100"/>
</p>

## Data Description

In this project, the training and testing data were provided by [**Kaggle competition-Diabetic Retinopathy Detection**](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).  

There are a total of 28100 images in the training data and a total of 7026 images in the testing data. In addition, there are a total of 5 categories in the dataset, corresponding to the severity of the five types of diabetes: No, Mild, Moderate, Severe and Proliferative DR.

The data used in this project can be obtained from Google driver, which is called [**data**](https://drive.google.com/drive/folders/1ENoGQj9W6B_4a4W_ajAJZU8K7o_5Tf8R?usp=sharing) directory. Just download the folder from Google driver and extract it into the **Diabetic-Retinopathy-Detection** folder.

## Training

A total of four methods are provided in this project to train the model, corresponding to the four files **ResNet18_nonpretrained_model.py**, **ResNet18_pretrained_model.py**, **ResNet50_nonpretrained_model.py** and **ResNet50_pretrained_model.py**.
