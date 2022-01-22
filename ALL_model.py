from torchvision import models
import torch.nn as nn



class ResNet18_pretrained(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18_pretrained, self).__init__()

        self.classify = nn.Linear(512, 5)
        pretrained_model = models.__dict__['resnet{}'.format(18)](pretrained=True)
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

class ResNet50_pretrained(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_pretrained, self).__init__()

        self.classify = nn.Linear(2048, 5)
        pretrained_model = models.__dict__['resnet{}'.format(50)](pretrained=True)
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

class ResNet18_nonpretrained(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18_nonpretrained, self).__init__()


        self.classify = nn.Linear(512, 5)
        pretrained_model = models.__dict__['resnet{}'.format(18)](pretrained=False)
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

class ResNet50_nonpretrained(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_nonpretrained, self).__init__()


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


def ResNet18_pretrained_return_model():

    model = ResNet18_pretrained()

    for name,child in model.named_children():
        if name in ['layer4','fc']:
            #print(name + 'is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            #print(name + 'is frozen')
            for param in child.parameters():
                param.requires_grad = False

    return model

def ResNet50_pretrained_return_model():

    model = ResNet50_pretrained()

    for name,child in model.named_children():
        if name in ['layer4','fc']:
            #print(name + 'is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            #print(name + 'is frozen')
            for param in child.parameters():
                param.requires_grad = False

    return model

def ResNet18_nonpretrained_return_model():

    model = ResNet18_nonpretrained()

    model_layer = [model.layer1,model.layer2,model.layer3,model.layer4]
    for layer in model_layer:
        for number in range(len(layer)):
            layer[number].relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)    

    model.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)

    return model

def ResNet50_nonpretrained_return_model():

    model = ResNet50_nonpretrained()

    model_layer = [model.layer1,model.layer2,model.layer3,model.layer4]
    for layer in model_layer:
        for number in range(len(layer)):
            layer[number].relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)    

    model.relu = nn.LeakyReLU(negative_slope=0.01,inplace=True)

    return model


