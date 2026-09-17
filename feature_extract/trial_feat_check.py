import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import cv2
import os
from sobel_loss_feature_ex import edge_loss

class Feature(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.encode=nn.Sequential(
            # layer 1
            nn.Conv2d(in_channels=in_channel,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # layer 2
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #layer 3
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.flat=nn.Flatten()
        self.global_pooling=nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        x=self.encode(x)
        x=self.global_pooling(x)
        x=self.flat(x)
        x=nn.functional.normalize(
            x,p=2,dim=1
        )
        # print(x.shape)
        return x

model=Feature()
model.load_state_dict(torch.load("feature_ex_V0.pth",map_location="cpu"))
lr=r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_LR_bicubic\X2\0015x2.png"
hr=r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_HR\0001.png"
transform=transforms.ToTensor()

lr_image=cv2.imread(lr)
lr_image=cv2.resize(lr_image,(840,480),interpolation=cv2.INTER_AREA)
lr_image=cv2.cvtColor(lr_image,cv2.COLOR_BGR2RGB)
lr_tensor=transform(lr_image).unsqueeze(0)

hr_image=cv2.imread(hr)
hr_image=cv2.resize(hr_image,(1280,720),interpolation=cv2.INTER_AREA)
hr_image=cv2.cvtColor(hr_image,cv2.COLOR_BGR2RGB)
hr_tensor=transform(hr_image).unsqueeze(0)

critieon=nn.CosineSimilarity(dim=1)
model.eval()

lr_feature=model(lr_tensor)
hr_feature=model(hr_tensor)

loss=critieon(lr_feature,hr_feature)
sobel_loss=edge_loss(lr_tensor,hr_tensor)

print("Edge loss:",sobel_loss.item(),"cosine similarity:",loss.item())