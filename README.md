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
$ conda create --name retinopathyenv python=3.8 -y
$ conda activate retinopathyenv
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

The data used in this project can be obtained from Google driver, which is called [**data**](https://drive.google.com/file/d/1GxKDAO-KpP8NfX0wNermBiFW9EKuSuG3/view?usp=sharing) directory. Just download the folder from Google driver and extract it into the **Diabetic-Retinopathy-Detection** folder.

## Directory Tree

In this project, you can put the folder on the specified path according to the pattern in the following directory tree for training and testing.

```bash=
├─ ALL_model.py
├─ data
│  ├─ 10003_left.jpeg
│  ├─ 10003_right.jpeg
│  ├─ 10007_left.jpeg
│  ├─ 10007_right.jpeg
│  ├─ 10009_left.jpeg
│  ├─ 10009_right.jpeg
│  ├─ 1000_left.jpeg
│  ├─ ...
│  ├─ ...
│  └─ 99_right.jpeg
├─ dataloader.py
├─ history_csv
│  ├─ ResNet18_nonpretrained.csv
│  ├─ ResNet18_pretrained.csv
│  ├─ ResNet50_nonpretrained.csv
│  └─ ResNet50_pretrained.csv
├─ model_testing.py
├─ model_weight
│  ├─ ResNet18_nonpretrained.rar
│  ├─ ResNet18_pretrained.rar
│  ├─ ResNet50_nonpretrained.rar
│  └─ ResNet50_pretrained.rar
├─ Plot_History_Result.py
├─ ResNet18_nonpretrained_model.py
├─ ResNet18_pretrained_model.py
├─ ResNet50_nonpretrained_model.py
├─ ResNet50_pretrained_model.py
├─ test_img.csv
├─ test_label.csv
├─ train_img.csv
├─ train_label.csv
└─ README.md
```

## Training

A total of four methods are provided in this project to train the model, corresponding to the four files **ResNet18_nonpretrained_model.py**, **ResNet18_pretrained_model.py**, **ResNet50_nonpretrained_model.py** and **ResNet50_pretrained_model.py**.

You can get some detailed introduction and experimental results in the link below.  
https://github.com/secondlevel/Diabetic-Retinopathy-Detection/blob/main/Experiment%20Report.pdf

You can config the training parameters through the following argparse, and use the following instructions to train different model.  

```bash=
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default='10', help='training epochs')
parser.add_argument('--image_size', type=int, default='224', help='model input image size')
parser.add_argument('--n_channels', type=int, default='3', help='model input image channels')
parser.add_argument('--train_batch_size', type=int, default='256', help='batch size to training')
parser.add_argument('--test_batch_size', type=int, default='281', help='batch size to testing')
parser.add_argument('--number_worker', type=int, default='4', help='number worker')
parser.add_argument('--learning_rate', type=float, default='5e-3', help='learning rate')
parser.add_argument('--save_model', action='store_true', help='check if you want to save the model.')
parser.add_argument('--save_csv', action='store_true', help='check if you want to save the training history.')
opt = parser.parse_args()
```

- #### ResNet18 pretrained model

```bash=
python ResNet18_pretrained_model.py --epochs 10 --save_model --save_csv
```

- #### ResNet50 pretrained model

```bash=
python ResNet50_pretrained_model.py --epochs 10 --save_model --save_csv
```

- #### ResNet18 non-pretrained model

```bash=
python ResNet18_nonpretrained_model.py --epochs 5 --save_model --save_csv
```

- #### ResNet50 non-pretrained model

```bash=
python ResNet50_nonpretrained_model.py --epochs 5 --save_model --save_csv
```

## Testing

You can display the testing results in different models by using the following commands which contains pretrained and non-pretrained models. The model weight could be downloaded from the [link](https://drive.google.com/drive/folders/1lWmzqmpvHbNxTR7fGrkWl5c4tRgq70wH?usp=sharing).    

The detailed experimental result are in the following link.  
https://github.com/secondlevel/EEG-classification/blob/main/Experiment%20Report.pdf

Then you will get the best result like this, each of the values were the testing accuracy.  

|          | Pretrained | None-Pretrained |
|:----------:|:------------:|:-----------------:|
| ResNet18 | 0.758281   | 0.705448        |
| ResNet50 | 0.774843   | 0.702573        |

## Reference

- https://www.kaggle.com/c/diabetic-retinopathy-detection#description
- https://arxiv.org/pdf/1712.09913.pdf
- https://ieeexplore.ieee.org/document/7780459
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
