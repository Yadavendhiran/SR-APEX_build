import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import os

#  load dataset

hr_p=r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_HR"
lr_p=r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_LR_bicubic\X2"

hr_path=sorted(os.listdir(r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_HR"))
# low re - y
lr_path=sorted(os.listdir(r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_LR_bicubic\X2"))

#  ------------ naming file name -------------
# create empty dataset
# x_train=[]
# y_train=[]
# for hr_name,lr_name in zip(hr_path,lr_path):
#     x_train.append(hr_name)
#     y_train.append(lr_name)

# print(hr_p+x_train[0]," ",lr_p+y_train[0])
# ---------------------------------------------

# --------custom dataset------
class div2kdata(Dataset):
    def __init__(self,hr_path,lr_path):
        self.hr_p=hr_path
        self.lr_p=lr_path
        self.hr_images=sorted(os.listdir(self.hr_p))
        self.lr_images=sorted(os.listdir(self.lr_p))
        self.transform=transforms.ToTensor()

    def __len__(self): # --- len of hr-images
        return len(self.hr_images)
    
    def __getitem__(self, indx):
        hr=Image.open(os.path.join(self.hr_p,self.hr_images[indx])).convert("RGB")
        lr=Image.open(os.path.join(self.lr_p,self.lr_images[indx])).convert("RGB")
        #--resize of each image to stack in tensor data
        self.hr_resize=transforms.RandomCrop(size=(1000,1000))
        self.lr_resize=transforms.RandomCrop(size=(400,400))
        self.HR_resize=self.hr_resize(hr)
        self.LR_resize=self.lr_resize(lr)
        hr_tensor=self.transform(self.HR_resize)
        lr_tensor=self.transform(self.LR_resize)
        return lr_tensor,hr_tensor # x = lr , y = hr

dataload=div2kdata(hr_p,lr_p)
#--load dataloader--
tensor_load=DataLoader(dataset=dataload,batch_size=4,shuffle=True)

#---create CNN using pytorch

class CNNSR(nn.Module):
    def __init__(self,n_channel=3):
        