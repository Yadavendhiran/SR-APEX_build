from torchvision import transforms
import torch.nn as nn
import torch,cv2,os
import torch.nn.functional as F

lr_img_path=r"D:\yt\collegeProj\dataset video\train_480_sharp\000"
hr_image_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000"

lr_image_all=sorted(os.listdir(lr_img_path))
hr_image_all=sorted(os.listdir(hr_image_path))

trans=transforms.ToTensor()
device="cuda" if torch.cuda.is_available() else "cpu"

texture_high = torch.tensor([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=torch.float32).view(1,1,3,3)

texture_lap = torch.tensor([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=torch.float32).view(1,1,3,3)

# texture_diag1 = torch.tensor([
#     [-1, 0, -1],
#     [0, 4, 0],
#     [-1, 0, -1]
# ], dtype=torch.float32).view(1,1,3,3)

# texture_diag2 = torch.tensor([
#     [-1, 0, -1],
#     [0, 4, 0],
#     [-1, 0, -1]
# ], dtype=torch.float32).view(1,1,3,3) we had selected epxert 1 and 2 as experts

experts=[texture_high,texture_lap]

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

class apex_texture(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_1=nn.Parameter(texture_high,requires_grad=True)
        self.expert_2=nn.Parameter(texture_lap,requires_grad=True)

    def forward(self,x):
        exp_1=F.conv2d(x,self.expert_1,padding=1)
        exp_2=F.conv2d(x,self.expert_2,padding=1)

        exp_c=torch.cat([exp_1,exp_2],dim=1)
        return exp_c
    
model=apex_texture().to(device)
criteon=nn.MSELoss()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

print("initial:")
print(model.expert_1.detach().cpu())
print(model.expert_2.detach().cpu())
for i in range(len(hr_tensors)):
    optim.zero_grad()
    lr_feature=model(lr_tensors[i].to(device,non_blocking=True))
    hr_feature=model(hr_tensors[i].to(device,non_blocking=True))
    loss=criteon(lr_feature,hr_feature)
    loss.backward()
    optim.step()
    print("loss per file: ",loss.item())

print("final:")
print(model.expert_1.detach().cpu())
print(model.expert_2.detach().cpu())

image_read=cv2.imread(r"D:\yt\collegeProj\dataset video\train_480_sharp\001\00000000.png")
image_read=cv2.cvtColor(image_read,cv2.COLOR_BGR2RGB)
image_read=trans(image_read).unsqueeze(0).to(device,non_blocking=True)
hr=model(image_read.mean(dim=1,keepdim=True)).to(device)
print(hr.detach().cpu().numpy())
print("Shape after hr testing:")
print(model.expert_1.detach().cpu())
print(model.expert_2.detach().cpu())