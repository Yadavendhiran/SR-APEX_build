import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import cv2
import os

lr_path=r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_LR_bicubic\X2"
hr_path=r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_HR"
device="cuda"

# sobel loss
sobelx=torch.tensor([[-1,-2,-1],
         [0,0,0],
         [1,2,1]],
         dtype=torch.float32).view(1,1,3,3).to(device)

sobley=torch.tensor([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]],dtype=torch.float32).view(1,1,3,3).to(device)

def edge_loss(predict_lr,true_hr):
    predict_image=predict_lr.mean(dim=1,keepdim=True)
    true_hr_image=true_hr.mean(dim=1,keepdim=True)
    predict_image=nn.functional.interpolate(
        predict_image,size=true_hr_image.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    predicted_x=nn.functional.conv2d(
        predict_image,sobelx,padding=1
    )
    predicted_y=nn.functional.conv2d(
        predict_image,sobley,padding=1
    )
    true_x=nn.functional.conv2d(
        true_hr_image,sobelx,padding=1
    )
    true_y=nn.functional.conv2d(
        true_hr_image,sobley,padding=1
    )

    loss_x=nn.functional.l1_loss(predicted_x,true_x)
    loss_y=nn.functional.l1_loss(predicted_y,true_y)

    return loss_x+loss_y

# assuming folder-> direct image files
class DIV2Kdataset(Dataset):
    def __init__(self,lr_path,hr_path):
        self.lr_path=lr_path
        self.hr_path=hr_path
        self.lr_all=sorted(os.listdir(self.lr_path))  # resize to 854,480
        self.hr_all=sorted(os.listdir(self.hr_path))  # resize to 1280,720
        self.transform=transforms.ToTensor()

    def __len__(self):
        return len(self.lr_all)

    def __getitem__(self, index):
        lr_image_path=os.path.join(lr_path,self.lr_all[index])  # dir foler + file combined for lr
        lr_image=cv2.imread(lr_image_path)
        lr_image=cv2.resize(lr_image,(854,480),interpolation=cv2.INTER_AREA) # 480 convertion
        lr_image=cv2.cvtColor(lr_image,cv2.COLOR_BGR2RGB)                    # change color bgr to rgb
        hr_image_path=os.path.join(hr_path,self.hr_all[index]) # dir folder + file combined for hr
        hr_image=cv2.imread(hr_image_path)
        hr_image=cv2.resize(hr_image,(1280,720),interpolation=cv2.INTER_AREA) # 720 convertion
        hr_image=cv2.cvtColor(hr_image,cv2.COLOR_BGR2RGB)  # change color bgr to rgb

        lr_tensor=self.transform(lr_image)
        hr_tensor=self.transform(hr_image)

        return lr_tensor,hr_tensor

data=DIV2Kdataset(lr_path,hr_path)
dataset=DataLoader(data,batch_size=4,num_workers=0)

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

model=Feature().to(device)
critieon=nn.CosineSimilarity()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

epochs=20
edge_weight=0.1
total_loss=[]
for i in range(epochs):
    cosine_loss=0
    file=0
    sobel_loss=0
    combine_loss=0
    for lr_tensor,hr_tensor in dataset:
        lr=lr_tensor.to(device,non_blocking=True)
        hr=hr_tensor.to(device,non_blocking=True)

        lr_feature=model(lr)  # return Low Feature
        hr_feature=model(hr)  # return high feature

        loss=1-critieon(lr_feature,hr_feature).mean() # comapre lr and hr features
        optim.zero_grad()
        cosine_loss+=loss.item()
        sobels=edge_loss(lr,hr)
        sobel_loss+=sobels.item()
        all_loss=loss+(sobels*edge_weight)
        all_loss.backward()
        optim.step()
        combine_loss=cosine_loss+sobel_loss
        file+=1

    average_cosine_loss=cosine_loss/file
    average_sobel_loss=sobel_loss/file
    average_combine_loss=combine_loss/file
    total_loss.append(average_combine_loss)
    print("epochs:",i+1,"cosine loss:",average_cosine_loss,"sobel loss:",average_sobel_loss)


torch.save(model.state_dict(),"feature_ex_V0.pth")