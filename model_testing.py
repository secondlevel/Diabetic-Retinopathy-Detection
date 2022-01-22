from ALL_model import ResNet18_pretrained_return_model, ResNet18_nonpretrained_return_model, ResNet50_pretrained_return_model, ResNet50_nonpretrained_return_model
from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import pandas as pd
import torch
import numpy 
from tqdm import tqdm
import os

def testing_batch_accuracy(test_loader,model,device):

    all_test_accuracy = []
    model.eval()
    with torch.no_grad():

        model.cuda(0)
        for x_test,y_test in tqdm(test_loader):
        
            n = len(x_test)
            # y_true_All_test_batch+= (y_test.numpy().tolist())

            # y_true_All_test_batch+=y_test
            
            x_test, y_test = x_test.to(device), y_test.to(device)
            y_pred_test = model(x_test)

            correct_test = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
            # y_pred_All_test_batch += (torch.max(y_pred_test,1)[1].cpu().numpy().tolist())
            # y_pred_All_test_batch += torch.max(y_pred_test,1)[1]

            all_test_accuracy.append(correct_test/n)

            # print("testing accuracy:",correct/n)
        
    return sum(all_test_accuracy)/len(all_test_accuracy)

if __name__ == "__main__":

    model_list=[ResNet18_pretrained_return_model, ResNet18_nonpretrained_return_model, ResNet50_pretrained_return_model, ResNet50_nonpretrained_return_model]
    model_file_path=["ResNet18_pretrained.rar", "ResNet18_nonpretrained.rar", "ResNet50_pretrained.rar", "ResNet50_nonpretrained.rar"]

    data_path = os.path.dirname(os.path.abspath(__file__))+"/data/"

    test_transform = transforms.Compose
    ([
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        transforms.Resize((224,224))
    ])

    test_dataset = RetinopathyLoader(data_path, "test", transform=test_transform)
    test_loader = DataLoader(test_dataset,batch_size=281,num_workers=4)
    print()

    Pretrained=[]
    Non_Pretrained=[]

    for i in range(len(model_list)):


        filepath=os.path.abspath(os.path.dirname(__file__))+"\\model_weight\\"+model_file_path[i]
        
        device = torch.device("cuda:0")
        model = model_list[i]()
        model.load_state_dict(torch.load(filepath))
        
        # print(model)
        # summary(model.cuda(),(3,224,224))
        print(model_file_path[i][:-4],"testing")
        print("-----------------------------------------------------------------------------")
        testing_accuracy = testing_batch_accuracy(test_loader,model,device)
        print()

        if "nonpretrained" in model_file_path[i]:
            Non_Pretrained.append(testing_accuracy)
        elif "pretrained" in model_file_path[i]:
            Pretrained.append(testing_accuracy)

    df = pd.DataFrame({"Pretrained":Pretrained,"None-Pretrained":Non_Pretrained},index=["ResNet18","ResNet50"])
    print(df)