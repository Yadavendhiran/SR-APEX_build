from srcnn_180_720 import SRCNN
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import os
from matplotlib import pyplot as plt

device="cuda"

flie_list={}

lr_180_path=r"D:\yt\collegeProj\dataset video\train_sharp_bicubic\train\train_sharp_bicubic\X4"
hr_720_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp"

all_lr_file=sorted(os.listdir(lr_180_path)) # eg 000,001,002,...
all_720_file=sorted(os.listdir(hr_720_path)) # eg 000,001,002,...

for i in range(11):
    flie_list[all_720_file[i]]=all_lr_file[i]

print(len(flie_list))

class ImageLoader(Dataset): # 1080->720 = hr , 480->18o=lr
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

model=SRCNN().to(device)
crition=nn.MSELoss()
print(flie_list)
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
epochs=5
total_loss=[]
total_file=0
for i in range(epochs):
    mse_loss_epochs=0
    average_mse_loss_epochs=0
    for key,value in flie_list.items():
        dataset=ImageLoader(key,hr_720_path,value,lr_180_path)
        data=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0)
        average_mse_loss=0
        mse_loss=0
        file=0
        for lr,hr in data:
            lr=lr.to(device,non_blocking=True)
            hr=hr.to(device,non_blocking=True)
            lr_feat_180=model(lr)
            loss=crition(lr_feat_180,hr)
            optim.zero_grad()
            loss.backward()
            optim.step()
            mse_loss+=loss.item()
            mse_loss_epochs+=loss.item()
            file+=1

        average_mse_loss=mse_loss/(len(data)*file)
        total_loss.append(average_mse_loss)
        total_file+=1 # folder count
        print("file no:",key,"average mse loss:",average_mse_loss)  # file loss mse total 99 files and dataset len
    average_mse_loss_epochs=mse_loss_epochs/(len(data)*len(flie_list))     # folde wise loss mse total 11 or X folder * len dataset = len data
    print("epochs:",i+1,"average:",average_mse_loss_epochs)

x=total_loss
y=list(range(1,total_file+1))
plt.plot(x,y)
plt.xlabel("loss per folder")
plt.ylabel("folders")
plt.show()

torch.save(model.state_dict(),"SRCNN_4X.V1.pth")