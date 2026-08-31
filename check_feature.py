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

image_1=Image.open(r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_valid_HR\0801.png").convert("RGB")

image_2=Image.open(r"D:\yt\collegeProj\dataset img\Dataset\DIV2K_train_HR\0014.png").convert("RGB")

# crop image
crop_image=transforms.CenterCrop((400,400))
cropped_image_1=crop_image(image_1)
cropped_image_2=crop_image(image_2)

# convert to tensor
transform=transforms.ToTensor()
tensor_image_1=transform(cropped_image_1).unsqueeze(0).to(device)
tensor_image_2=transform(cropped_image_2).unsqueeze(0).to(device)

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
model.load_state_dict(
    torch.load(
        r"feature_extract.pth",
        map_location=device
    )
)
model.eval()

with torch.no_grad():
    feature_1=model(tensor_image_1)
    feature_2=model(tensor_image_2)

postivite_similar=F.cosine_similarity(
    feature_1,feature_2,dim=1
)

print("shape:",feature_1.shape," ",feature_2.shape)
print("Cosine_similarity:",postivite_similar.item())
