import torch
from torchvision import transforms
import cv2,os
import torch.nn as nn
import torch.nn.functional as F

lr_img_path=r"D:\yt\collegeProj\dataset video\train_480_sharp\000"
hr_image_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000"

lr_image_all=sorted(os.listdir(lr_img_path))
hr_image_all=sorted(os.listdir(hr_image_path))

trans=transforms.ToTensor()
device="cuda" if torch.cuda.is_available() else "cpu"

sobel_y=torch.tensor([[-1,-2,-1],
         [0,0,0],
         [1,2,1]],dtype=torch.float32).view(1,1,3,3).to(device)

sobel_x=torch.tensor(
    [[-1,0,1],
    [-2,0,2],
    [-1,0,1]],dtype=torch.float32
).view(1,1,3,3).to(device)

diagonal_left=torch.tensor(
    [
        [ 0,1, 2],
        [-1,0,1],
        [-2,-1,0]
    ],dtype=torch.float32
).view(1,1,3,3).to(device)

diagonal_right=torch.tensor(
    [
        [ 2 ,1 ,0],
        [ 1 ,0,-1],
        [ 0,-1,-2]
    ],dtype=torch.float32
).view(1,1,3,3).to(device)

experts=[sobel_x,sobel_y,diagonal_left,diagonal_right]

# epochs=2
# for i in range(epochs):
#     predict_sobelx=nn.functional.conv2d(gray_image,sobel_x,padding=1)

#     predict_sobely=nn.functional.conv2d(gray_image,sobel_y,padding=1)

#     predict_diagonal_left=nn.functional.conv2d(gray_image,diagonal_left,padding=1)

#     predict_diagonal_right=nn.functional.conv2d(gray_image,diagonal_right,padding=1)


class apex_base(nn.Module):
    def __init__(self):
        super().__init__()

        self.sobelx=nn.Parameter(sobel_x,requires_grad=True)
        self.relu1=nn.ReLU(inplace=True)
        self.sobely=nn.Parameter(sobel_y,requires_grad=True)
        self.relu2=nn.ReLU(inplace=True)
        self.diagonal_left=nn.Parameter(diagonal_left,requires_grad=True)
        self.relu3=nn.ReLU(inplace=True)
        self.diagonal_right=nn.Parameter(diagonal_right,requires_grad=True)
        self.relu4=nn.ReLU(inplace=True)
    
    def forward(self,x):
        x=x.mean(dim=1,keepdim=True)
        vertical=F.conv2d(x,self.sobelx,padding=1)
        horizontal=F.conv2d(x,self.sobely,padding=1)
        diagonal_l=F.conv2d(x,self.diagonal_left,padding=1)
        diagonal_r=F.conv2d(x,self.diagonal_right,padding=1)

        feature=torch.cat(
            [
                vertical,horizontal,diagonal_l,diagonal_r
            ],dim=1
        )

        return feature


lr_tensors=[]
hr_tensors=[]
for i in range(len(hr_image_all)):
    hr_img=cv2.imread(os.path.join(hr_image_path,hr_image_all[i]))
    lr_img=cv2.imread(os.path.join(lr_img_path,lr_image_all[i]))
    
    lr_image=cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
    lr_tensor=trans(lr_image).unsqueeze(0)

    hr_image=cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
    hr_image=cv2.resize(hr_image,(854,480),interpolation=cv2.INTER_AREA)
    hr_tensor=trans(hr_image).unsqueeze(0)

    lr_tensors.append(lr_tensor)
    hr_tensors.append(hr_tensor)

model=apex_base().to(device)
criteon=nn.MSELoss()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

print("initial:")
print(model.sobelx.detach().cpu())
print(model.sobely.detach().cpu())
print(model.diagonal_left.detach().cpu())
print(model.diagonal_right.detach().cpu())
for i in range(len(hr_tensors)):
    optim.zero_grad()
    lr_feature=model(lr_tensors[i].to(device,non_blocking=True))
    hr_feature=model(hr_tensors[i].to(device,non_blocking=True))
    loss=criteon(lr_feature,hr_feature)
    loss.backward()
    optim.step()
    print("loss per file: ",loss.item())

print("final:")
print(model.sobelx.detach().cpu())
print(model.sobely.detach().cpu())
print(model.diagonal_left.detach().cpu())
print(model.diagonal_right.detach().cpu())

image_read=cv2.imread(r"D:\yt\collegeProj\dataset video\train_480_sharp\001\00000000.png")
image_read=cv2.cvtColor(image_read,cv2.COLOR_BGR2RGB)
image_read=trans(image_read).unsqueeze(0).to(device,non_blocking=True)
hr=model(image_read).to(device)
print(hr)
print("Shape after hr testing:")
print(model.sobelx.detach().cpu())
print(model.sobely.detach().cpu())
print(model.diagonal_left.detach().cpu())
print(model.diagonal_right.detach().cpu())
