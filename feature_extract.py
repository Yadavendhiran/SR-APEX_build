import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

class DIVLOAD(Dataset):
    def __init__(self,image_path):
        self.image_p=image_path
        self.all_images=sorted(os.listdir(image_path))
        self.transform=transforms.ToTensor()

    def __len__(self):
        return (len(self.all_images))

    def __getitem__(self, index):
        images=Image.open(
            os.path.join(self.image_p,self.all_images[index])
        ).convert("RGB")
        image_c=transforms.CenterCrop((1920,1080))
        image_cropped=image_c(images)
        tensor_image=self.transform(image_cropped).to(device)

        return tensor_image

image_path=r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_HR"
dataset=DIVLOAD(image_path=image_path)
dataload=DataLoader(dataset,batch_size=4,shuffle=True,num_workers=0)

class feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=nn.Sequential(
            # layer 1
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #layer 2
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            # layer 3
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.flat=nn.Flatten()

    def forward(self,x):
        x=self.encode(x)
        x=self.global_pool(x)
        x=self.flat(x)
        x=F.normalize(
            x,p=2,dim=1
        )

        return x

model=feature().to(device)

model.train()
for image in dataload:
    images_data=model(image)

print("model trained")

torch.save(
    model.state_dict(),
    "feature_extract.pth"
)