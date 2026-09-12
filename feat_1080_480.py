from torch.utils.data import DataLoader,Dataset
import torch
import cv2
import os
from torchvision import transforms
import torch.nn as nn

folder_1080_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp"
folder_480_path=r"D:\yt\collegeProj\dataset video\train_480_sharp"

folder_1080=sorted(os.listdir(folder_1080_path))
folder_480=sorted(os.listdir(folder_480_path))
file_pair={}

for i in range(len(folder_480)):
    file_pair[folder_1080[i]]=folder_480[i]

print((file_pair))

device="cuda"

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
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            # layer 1
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            # layer 2
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            # layer 3
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            # layer 4
            nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            # layer 5
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(720,1280),mode="bilinear",align_corners=False),
            # layer 6
            nn.Conv2d(in_channels=16,out_channels=3,kernel_size=3,stride=1,padding=1),
        )

    def forward(self,x):
        x=self.encoder(x)
        return x

model=SRCNN().to(device)
crierion=nn.MSELoss()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
epchos=2
for i in range(epchos):
    total_loss_epcho=0
    average_loss_epcho=0
    for key,value in file_pair.items():
        total_loss_file=0
        average_loss_file=0
        data=ImageLoader(key,folder_1080_path,value,folder_480_path)
        dataset=DataLoader(data,batch_size=4,shuffle=True,num_workers=0)
        for lr_480,hr_1080 in dataset:
            lr_480=lr_480.to(device,non_blocking=True)
            hr_1080=hr_1080.to(device,non_blocking=True)
            predicted_hr=model(lr_480)
            loss=crierion(predicted_hr,hr_1080)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss_file+=loss.item()
            total_loss_epcho+=loss.item()
        average_loss_file=total_loss_file/len(dataset)
        print("PROCESSED FILE:",key,"Epcho:",i+1,"average loss:",average_loss_file)
    print("Epcho:",i+1,"average epcho loss:",(total_loss_epcho/(11*len(dataset))))

torch.save(model.state_dict(),"SRCNN_1.5X.pth")