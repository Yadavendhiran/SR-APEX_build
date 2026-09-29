import torch
from torchvision import transforms
import cv2,os
import torch.nn.functional as F
import torch.nn as nn

# integrate gradient with layer 1 the edges so layer 1 = edge + gradient

lr_img_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000"
hr_image_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000"

lr_image_all=sorted(os.listdir(lr_img_path))
hr_image_all=sorted(os.listdir(hr_image_path))

trans=transforms.ToTensor()
device="cuda" if torch.cuda.is_available() else "cpu"

gradient_x = torch.tensor([
    [-3,  0,  3],
    [-10, 0, 10],
    [-3,  0,  3]
], dtype=torch.float32).view(1,1,3,3)

gradient_y = torch.tensor([
    [-3, -10, -3],
    [ 0,   0,  0],
    [ 3,  10,  3]
], dtype=torch.float32).view(1,1,3,3)

gradient_diag1 = torch.tensor([
    [-10, -3,  0],
    [ -3,  0,  3],
    [  0,  3, 10]
], dtype=torch.float32).view(1,1,3,3)

gradient_diag2 = torch.tensor([
    [ 0,  3, 10],
    [-3,  0,  3],
    [-10,-3,  0]
], dtype=torch.float32).view(1,1,3,3)

experts=[gradient_x,gradient_y,gradient_diag1,gradient_diag2]

lr_tensors=[]
hr_tensors=[]
for i in range(len(hr_image_all)):
    hr_img=cv2.imread(os.path.join(hr_image_path,hr_image_all[i]))
    lr_img=cv2.imread(os.path.join(lr_img_path,lr_image_all[i]))
    
    lr_image=cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
    lr_tensor=trans(lr_image).unsqueeze(0)
    
    hr_image=cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
    # hr_image=cv2.resize(hr_image,(854,480),interpolation=cv2.INTER_AREA)
    hr_tensor=trans(hr_image).unsqueeze(0)

    lr_tensors.append(lr_tensor.mean(dim=1,keepdim=True))
    hr_tensors.append(hr_tensor.mean(dim=1,keepdim=True))

class apex_gradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_x=nn.Parameter(gradient_x,requires_grad=True)
        self.grad_y=nn.Parameter(gradient_y,requires_grad=True)
        self.grad_dig1=nn.Parameter(gradient_diag1,requires_grad=True)
        self.grad_dig2=nn.Parameter(gradient_diag2,requires_grad=True)

    def forward(self,x):
        grad_x=F.conv2d(x.to(device,non_blocking=True),self.grad_x.to(device,non_blocking=True),padding=1)
        grad_y=F.conv2d(x.to(device,non_blocking=True),self.grad_y.to(device,non_blocking=True),padding=1)
        grad_dig1=F.conv2d(x.to(device,non_blocking=True),self.grad_dig1.to(device,non_blocking=True),padding=1)
        grad_dig2=F.conv2d(x.to(device,non_blocking=True),self.grad_dig2.to(device,non_blocking=True),padding=1)

        feature=torch.cat(
            [grad_x,grad_y,grad_dig1,grad_dig2],dim=1
        )

        return feature

model=apex_gradient().to(device)
criteon=nn.MSELoss()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

print("initial:")
print(model.grad_x.detach().cpu())
print(model.grad_y.detach().cpu())
print(model.grad_dig1.detach().cpu())
print(model.grad_dig2.detach().cpu())

for lr,hr in zip(lr_tensors,hr_tensors):
    lr_input=lr.to(device,non_blocking=True)
    hr_input=hr.to(device,non_blocking=True)
    optim.zero_grad()
    feature_lr=model(lr_input)
    feature_hr=model(hr_input)
    loss=criteon(feature_lr,feature_hr)
    loss.backward()
    optim.step()
    print("loss per file:",loss.item())

print("final:")
print(model.grad_x.detach().cpu())
print(model.grad_y.detach().cpu())
print(model.grad_dig1.detach().cpu())
print(model.grad_dig2.detach().cpu())

image_read=cv2.imread(r"D:\yt\collegeProj\dataset video\train_480_sharp\001\00000000.png")
image_read=cv2.cvtColor(image_read,cv2.COLOR_BGR2RGB)
image_read=trans(image_read).unsqueeze(0).to(device,non_blocking=True)
image_read=image_read.mean(dim=1,keepdim=True)
hr=model(image_read).to(device)
print(hr)
print("Shape after hr testing:")
print(model.grad_x.detach().cpu())
print(model.grad_y.detach().cpu())
print(model.grad_dig1.detach().cpu())
print(model.grad_dig2.detach().cpu())
# lr_obvs=[]
# hr_obvs=[]
# for i in experts:
#     lr_result=[]
#     hr_result=[]
#     for lr,hr in zip(lr_tensors,hr_tensors):
#         lr_r=F.conv2d(lr.to(device,non_blocking=True),i.to(device,non_blocking=True),padding=1)
#         hr_r=F.conv2d(hr.to(device,non_blocking=True),i.to(device,non_blocking=True),padding=1)
#         lr_result.append(lr_r.detach().cpu().numpy())
#         hr_result.append(hr_r.detach().cpu().numpy())   
#     lr_obvs.append(lr_result)
#     hr_obvs.append(hr_result)

# sum_lr=[]
# for i in range(len(lr_obvs)):
#     sum_lr_exp=0
#     for j in range(len(lr_obvs[i])):
#         sum_lr_exp+=lr_obvs[i][j].sum()
#     sum_lr.append(sum_lr_exp)
# print("sum of lr :",sum_lr)

# sum_hr=[]
# for i in range(len(hr_obvs)):
#     sum_hr_exp=0
#     for j in range(len(hr_obvs[i])):
#         sum_hr_exp+=hr_obvs[i][j].sum()
#     sum_hr.append(sum_hr_exp)
# print("sum of hr :",sum_hr)
