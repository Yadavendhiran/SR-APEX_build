from torchvision import transforms
import torch.nn as nn
import torch,cv2,os
import torch.nn.functional as F

# hypotesis use tensor 1,3,5

lr_img_path=r"D:\yt\collegeProj\dataset video\train_480_sharp\000"
hr_image_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000"

lr_image_all=sorted(os.listdir(lr_img_path))
hr_image_all=sorted(os.listdir(hr_image_path))

trans=transforms.ToTensor()
device="cuda" if torch.cuda.is_available() else "cpu"

noise_high=torch.tensor(
    [
        [-1,-1,-1],
        [-1 ,8,-1],
        [-1,-1,-1]
        ],dtype=torch.float32
).view(1,1,3,3)

# noise_lap=torch.tensor(
#     [
#         [ 0,-1,0],
#         [-1,4,-1],
#         [0 ,-1 ,0]
#     ],dtype=torch.float32
# ).view(1,1,3,3)

noise_fine_horizontal=torch.tensor(
    [
        [1 ,-2  ,1],
        [1 ,-2  ,1],
        [1 ,-2  ,1]
    ],dtype=torch.float32
).view(1,1,3,3)

# noise_fine_vertical=torch.tensor(
#     [
#      [1 ,1, 1],
#      [-2 ,-2,-2],
#      [ 1 ,1, 1]  
#      ],dtype=torch.float32
# ).view(1,1,3,3)

noise_digonal_horizontal=torch.tensor(
    [
     [1 ,0, -1],
     [0 ,-2, 0],
     [-1 ,0, 1]
    ],dtype=torch.float32
).view(1,1,3,3)

# noise_digonal_local=torch.tensor(
#     [
#      [-1 ,2, -1],
#      [ 2 ,-4, 2],
#      [-1 ,2, -1]
#     ],dtype=torch.float32
# ).view(1,1,3,3)

# experts=[noise_high,noise_lap,noise_fine_horizontal,noise_fine_vertical,noise_digonal_horizontal,noise_digonal_local]

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

    lr_tensors.append(lr_tensor.mean(dim=1,keepdim=True))
    hr_tensors.append(hr_tensor.mean(dim=1,keepdim=True))

class apex_noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_high=nn.Parameter(noise_high,requires_grad=True)
        self.noise_lap=nn.Parameter(noise_fine_horizontal,requires_grad=True)
        self.noise_dig_hori=nn.Parameter(noise_digonal_horizontal,requires_grad=True)

    def forward(self,x):
        
        noise_h=F.conv2d(x,self.noise_high.to(device,non_blocking=True),padding=1)
        noise_l=F.conv2d(x,self.noise_lap.to(device,non_blocking=True),padding=1)
        noise_d_h=F.conv2d(x,self.noise_dig_hori.to(device,non_blocking=True),padding=1)

        feature=torch.cat(
            [noise_h,noise_l,noise_d_h],dim=1
        )
        return feature 

model=apex_noise().to(device)
criteon=nn.MSELoss()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

print("initial:")
print(model.noise_high.detach().cpu())
print(model.noise_lap.detach().cpu())
print(model.noise_dig_hori.detach().cpu())

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
print(model.noise_high.detach().cpu())
print(model.noise_lap.detach().cpu())
print(model.noise_dig_hori.detach().cpu())

image_read=cv2.imread(r"D:\yt\collegeProj\dataset video\train_480_sharp\001\00000000.png")
image_read=cv2.cvtColor(image_read,cv2.COLOR_BGR2RGB)
image_read=trans(image_read).unsqueeze(0).to(device,non_blocking=True)
image_read=image_read.mean(dim=1,keepdim=True)
hr=model(image_read).to(device)
print(hr)
print("Shape after hr testing:")
print(model.noise_high.detach().cpu())
print(model.noise_lap.detach().cpu())
print(model.noise_dig_hori.detach().cpu())
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