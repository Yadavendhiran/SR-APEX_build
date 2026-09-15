from torch.utils.data import DataLoader,Dataset
import torch
import cv2
import os
from torchvision import transforms
import torch.nn as nn
from srcnn_180_720 import SRCNN
import matplotlib.pyplot as plt
import numpy as np

folder_1080_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp"
folder_480_path=r"D:\yt\collegeProj\dataset video\train_480_sharp"

folder_1080=sorted(os.listdir(folder_1080_path))
folder_480=sorted(os.listdir(folder_480_path))
file_pair={}

for i in range(12):
    file_pair[folder_1080[i]]=folder_480[i]

print((file_pair))

device="cuda"

sobelx=torch.tensor(
    [[-1.0,0.0,1.0],
     [-2.0,0.0,2.0],
     [-1.0,0.0,1.0]],dtype=torch.float32
).view(1,1,3,3).to(device)

sobely=torch.tensor(
    [[-1.0,-2.0,-1.0],
     [0.0,0.0,0.0],
     [1.0,2.0,1.0]],dtype=torch.float32
).view(1,1,3,3).to(device)


# edge loss using sobel
def edge_loss(predict_hr,true_hr):
    predicted_hr=predict_hr.mean(dim=1,keepdim=True)
    true_hr_gray=true_hr.mean(dim=1,keepdim=True)

    predicted_x=nn.functional.conv2d(
        predicted_hr,sobelx,padding=1
    )
    predicted_y=nn.functional.conv2d(
        predicted_hr,sobely,padding=1
    )

    true_hr_x=nn.functional.conv2d(
        true_hr_gray,sobelx,padding=1
    )
    true_hr_y=nn.functional.conv2d(
        true_hr_gray,sobely,padding=1
    )
    loss_x=nn.functional.l1_loss(predicted_x,true_hr_x)
    loss_y=nn.functional.l1_loss(predicted_y,true_hr_y)
    return loss_x+loss_y

class ImageLoader(Dataset):
    def __init__(self,path_1080_file,path_1080,path_480_file,path_480):
        self.path_1080=os.path.join(path_1080,path_1080_file) # D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000
        self.path_480=os.path.join(path_480,path_480_file) # D:\yt\collegeProj\dataset video\train_480_sharp\000
        self.lr_480_all=sorted(os.listdir(self.path_480)) # D:\yt\collegeProj\dataset video\train_480_sharp\000\00.png
        self.hr_1080_all=sorted(os.listdir(self.path_1080))

        self.transformer=transforms.ToTensor()

    def __len__(self):
        return (len(self.lr_480_all))

    def __getitem__(self, index):
        lr_480_image=cv2.imread(os.path.join(self.path_480,self.lr_480_all[index]))
        lr_480_image=cv2.cvtColor(lr_480_image,cv2.COLOR_BGR2RGB)
        hr_1080_image=cv2.imread(os.path.join(self.path_1080,self.hr_1080_all[index]))
        hr_1080_image=cv2.cvtColor(hr_1080_image,cv2.COLOR_BGR2RGB)

        lr_480_tensor=self.transformer(lr_480_image)
        # hr_1080_c=transforms.CenterCrop((1280,720))
        # hr_1080_cropped=hr_1080_c(hr_1080_image)
        hr_1080_tensor=self.transformer(hr_1080_image)

        return lr_480_tensor,hr_1080_tensor

# 1.5x SRCNN 

model=SRCNN().to(device)
crierion=nn.MSELoss()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
edge_weight=0.1
epchos=3
avg_epchho_loss=[]
avg_file_loss=[]
total_file=0
for i in range(epchos):
    total_loss_epcho=0
    average_loss_epcho=0
    total_mse_loss=0
    total_sobel_loss=0
    for key,value in file_pair.items():
        total_loss_file=0
        average_loss_file=0
        data=ImageLoader(key,folder_1080_path,value,folder_480_path)
        dataset=DataLoader(data,batch_size=4,shuffle=True,num_workers=0)
        for lr_480,hr_1080 in dataset:
            lr_480=lr_480.to(device,non_blocking=True)
            hr_1080=hr_1080.to(device,non_blocking=True)
            predicted_hr=model(lr_480)

            edge_loss_value=edge_loss(predicted_hr,hr_1080)
            total_sobel_loss+=edge_loss_value.item()
            loss=crierion(predicted_hr,hr_1080)
            total_mse_loss+=loss.item()
            total_loss=loss+(edge_weight*edge_loss_value)
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            total_loss_file+=total_loss.item()
            total_loss_epcho+=total_loss.item()


        average_loss_file=total_loss_file/len(dataset)
        average_mse_loss=total_mse_loss/len(dataset)
        average_sobel_loss=total_sobel_loss/len(dataset)

        avg_file_loss.append(average_loss_file)
        total_file+=1
        print("PROCESSED FILE:",key,"Epcho:",i+1,"average mse loss:",average_mse_loss,"average sobel loss:",average_sobel_loss)
    print("Epcho:",i+1,"average epcho mse loss:",(total_mse_loss/(len(file_pair)*len(dataset))),"average epchos sobel loss:",(total_sobel_loss/(len(file_pair)*len(dataset))))
    avg_epchho_loss.append(total_loss_epcho/(11*len(dataset)))

x=avg_file_loss
# print(len(x))
y=list(range(1,total_file+1))

plt.plot(x,y)
plt.xlabel("loss of folder")
plt.ylabel("No of folder")
plt.show()


torch.save(model.state_dict(),"SRCNN_1.5X_v2.pth")