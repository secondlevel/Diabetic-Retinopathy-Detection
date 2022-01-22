import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_loss_curve(ResNet18_pretrained,ResNet18_nonpretrained,ResNet50_pretrained,ResNet50_nonpretrained):

    plt.plot(ResNet18_pretrained['loss'], '-b', label='ResNet18_Pretrained')
    plt.plot(ResNet18_nonpretrained['loss'], '-g', label='ResNet18_Non-Pretrained')
    plt.plot(ResNet50_pretrained['loss'], '-r', label='ResNet50_Pretrained')
    plt.plot(ResNet50_nonpretrained['loss'], '-c', label='ResNet50_Non-Pretrained')

    plt.xlabel("Epoch",fontsize=13)
    plt.legend(loc='best')

    plt.ylabel("Loss Value",fontsize=13)
    plt.title("(Loss Curve)Pretrained Non-Pretrained comparision(All)",fontsize=18)

    plt.show()
    return "loss圖繪製成功"

def plot_ResNet18_accuracy_curve(ResNet18_pretrained,ResNet18_nonpretrained):

    plt.plot(np.array(ResNet18_pretrained['train_accuracy_history'])*100, '-b', label='Pretrained_train')
    plt.plot(np.array(ResNet18_nonpretrained['train_accuracy_history'])*100, '-g', label='Non-Pretrained_train')

    plt.plot(np.array(ResNet18_pretrained['test_accuracy_history'])*100, '-c', label='Pretrained_test')
    plt.plot(np.array(ResNet18_nonpretrained['test_accuracy_history'])*100, '-m', label='Non-Pretrained_test')

    plt.xlabel("Epoch",fontsize=13)
    plt.legend(loc='best')

    plt.ylabel("Accuracy(%)",fontsize=13)
    plt.title("(Accuracy Curve)Pretrained Non-Pretrained comparision(ResNet18)",fontsize=18)

    plt.show()
    return "ResNet18 Accuracy圖繪製成功"

def plot_ResNet50_accuracy_curve(ResNet50_pretrained,ResNet50_nonpretrained):

    plt.plot(np.array(ResNet50_pretrained['train_accuracy_history'])*100, '-b', label='Pretrained_train')
    plt.plot(np.array(ResNet50_nonpretrained['train_accuracy_history'])*100, '-g', label='Non-Pretrained_train')

    plt.plot(np.array(ResNet50_pretrained['test_accuracy_history'])*100, '-c', label='Pretrained_test')
    plt.plot(np.array(ResNet50_nonpretrained['test_accuracy_history'])*100, '-m', label='Non-Pretrained_test')

    plt.xlabel("Epoch",fontsize=13)
    plt.legend(loc='best')

    plt.ylabel("Accuracy(%)",fontsize=13)
    plt.title("(Accuracy Curve)Pretrained Non-Pretrained comparision(ResNet50)",fontsize=18)

    plt.show()
    return "DeepConvNet Accuracy圖繪製成功"

if __name__ == "__main__":
    
    path = os.path.abspath(os.path.dirname(__file__))+"/history_csv/"

    ResNet18_pretrained = pd.DataFrame(pd.read_csv(path+"ResNet18_pretrained.csv",encoding="utf-8-sig"))
    ResNet18_nonpretrained = pd.DataFrame(pd.read_csv(path+"ResNet18_nonpretrained.csv",encoding="utf-8-sig"))

    ResNet50_pretrained = pd.DataFrame(pd.read_csv(path+"ResNet50_pretrained.csv",encoding="utf-8-sig"))
    ResNet50_nonpretrained = pd.DataFrame(pd.read_csv(path+"ResNet50_nonpretrained.csv",encoding="utf-8-sig"))

    plot_loss_curve(ResNet18_pretrained,ResNet18_nonpretrained,ResNet50_pretrained,ResNet50_nonpretrained)
    # plot_ResNet18_accuracy_curve(ResNet18_pretrained,ResNet18_nonpretrained)
    # plot_ResNet50_accuracy_curve(ResNet50_pretrained,ResNet50_nonpretrained)

