from torchvision.transforms.transforms import RandomCrop
from dataloader import RetinopathyLoader
from torchvision import models
import torch.nn as nn
import torch
import os
import pkbar
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import transforms
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import itertools
import numpy as np
import argparse

def testing(y_pred_All_test_batch,y_true_All_test_batch,test_loader,model,device):

    # model.load_state_dict(torch.load(filepath))
    test_accuracy = []
    model.eval()
    with torch.no_grad():
        model.cuda(0)
        
        for x_test,y_test in tqdm(test_loader):
        
            n = len(x_test)
            y_true_All_test_batch+= (y_test.numpy().tolist())

            # y_true_All_test_batch+=y_test
            
            x_test,y_test = x_test.to(device),y_test.to(device)
            y_pred_test = model(x_test)

            correct_test = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
            y_pred_All_test_batch += (torch.max(y_pred_test,1)[1].cpu().numpy().tolist())
            # y_pred_All_test_batch += torch.max(y_pred_test,1)[1]

            test_accuracy.append(correct_test/n)

            # print("testing accuracy:",correct/n)
    
    test_accuracy = sum(test_accuracy)/len(test_accuracy)
        
    return y_pred_All_test_batch,y_true_All_test_batch,test_accuracy

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()


        self.classify = nn.Linear(2048, 5)
        pretrained_model = models.__dict__['resnet{}'.format(50)](pretrained=False)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # x = nn.Dropout(0.35)(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # x = nn.Dropout(0.35)(x)
        
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

def plot_confusion_matrix_figure(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_confusion_matrix(y_pred,y_true):
    # y_pred = [0,2,0,4,3,1,1,1,1,1]
    # y_true = [0,1,2,3,4,1,0,1,1,1]

    target_names = list(range(5))
    plt.figure()
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # print(cnf_matrix)
    plot_confusion_matrix_figure(cnf_matrix, classes=target_names,normalize=True,title='confusion matrix')
    # plt.savefig('confusion_matrix.jpg',dpi=300)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default='5', help='training epochs')
    parser.add_argument('--image_size', type=int, default='224', help='model input image size')
    parser.add_argument('--n_channels', type=int, default='3', help='model input image channels')
    parser.add_argument('--train_batch_size', type=int, default='64', help='batch size to training')
    parser.add_argument('--test_batch_size', type=int, default='281', help='batch size to testing')
    parser.add_argument('--number_worker', type=int, default='4', help='number worker')
    parser.add_argument('--learning_rate', type=float, default='5e-3', help='learning rate')
    parser.add_argument('--save_model', action='store_true', help='check if you want to save the model.')
    parser.add_argument('--save_csv', action='store_true', help='check if you want to save the training history.')
    opt = parser.parse_args()

    device = torch.device("cuda:0")
    path = os.path.dirname(os.path.abspath(__file__))+"/data/"
    epochs = opt.epochs
    lr = opt.learning_rate
    min_loss = 1
    max_accuracy = 0
    max_test_accuracy = 0

    filepath = os.path.abspath(os.path.dirname(__file__))+"\model_weight\ResNet50_nonpretrained.rar"
    filepath_csv = os.path.abspath(os.path.dirname(__file__))+"\history_csv\ResNet50_nonpretrained.csv"
    
    train_transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    # transforms.RandomCrop(224),
    transforms.Resize((opt.image_size,opt.image_size)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    transforms.Resize((opt.image_size,opt.image_size))
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = RetinopathyLoader(path,"train",transform=train_transform)
    test_dataset = RetinopathyLoader(path,"test",transform=test_transform)
    
    train_loader = DataLoader(train_dataset,batch_size=opt.train_batch_size,num_workers=opt.number_worker)
    test_loader = DataLoader(test_dataset,batch_size=opt.test_batch_size,num_workers=opt.number_worker)
    #1405
    #7025

    model = ResNet()

    # for name,child in model.named_children():
    #     if name in ['layer4','fc']:
    #         #print(name + 'is unfrozen')
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         #print(name + 'is frozen')
    #         for param in child.parameters():
    #             param.requires_grad = False

    model_layer = [model.layer1,model.layer2,model.layer3,model.layer4]
    for layer in model_layer:
        for number in range(len(layer)):
            layer[number].relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)    

    model.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)
    
    print(model)

    # model.to(device)
    model.cuda(0)
    summary(model.cuda(),(opt.n_channels,opt.image_size,opt.image_size))

    optimizer = optim.Adam(model.parameters(),lr = lr)
    # optimizer = optim.RMSprop(model.parameters(),lr = lr, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()

    loss_batch = []
    accuracy_batch = []
    loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    y_pred_All_test_batch = []
    y_true_All_test_batch = []

    for epoch in range(epochs):

        kbar = pkbar.Kbar(target=len(train_loader)-1, epoch=epoch, num_epochs=epochs, width=12, always_stateful=False)
        for i,(data, target) in enumerate(train_loader):
            model.train()
            data,target = data.to(device),target.to(device)

            # print("data.shape:",data.shape,"target.shape:",target.shape,"\n")
            y_pred = model(data)
            loss = criterion(y_pred, target)
            loss_batch.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss.item())

            n = target.shape[0]
            correct = (torch.max(y_pred,1)[1]==target).sum().item()
            train_accuracy = correct / n
            accuracy_batch.append(train_accuracy)
            kbar.update(i, values=[("loss", loss.item()), ("train accuracy", train_accuracy)])

        # print("\n epochs:",epoch,"loss:",sum(loss_batch)/len(loss_batch),"Training Accuracy:",sum(accuracy_batch)/len(accuracy_batch))
        y_pred_All_test_batch,y_true_All_test_batch,test_accuracy = testing(y_pred_All_test_batch,y_true_All_test_batch,test_loader,model,device)
        # kbar.add(1, values=[("testing accuracy",test_accuracy)])

        train_accuracy = sum(accuracy_batch)/len(accuracy_batch)
        train_loss = sum(loss_batch)/len(loss_batch)


        print("\n epochs:",epoch,"loss:",train_loss,"Training Accuracy:",train_accuracy,"Testing Accuracy:",test_accuracy)
        loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)
        
        loss_batch = []
        accuracy_batch = []

        # if train_loss<min_loss:
        #     min_loss = train_loss
            # torch.save(model.state_dict(), filepath)
        
        if train_accuracy>max_accuracy:
            max_accuracy = train_accuracy
            if opt.save_model: 
                torch.save(model.state_dict(), filepath)
    
    df = pd.DataFrame({"loss":loss_history,"train_accuracy_history":train_accuracy_history,"test_accuracy_history":test_accuracy_history})
    # print(df)
    if opt.save_csv: 
        df.to_csv(filepath_csv,encoding="utf-8-sig")
    plot_confusion_matrix(y_pred_All_test_batch,y_true_All_test_batch)
