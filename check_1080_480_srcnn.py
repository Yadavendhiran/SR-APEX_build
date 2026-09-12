import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np

device="cuda"

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
    
test_image=cv2.imread(r"D:\yt\collegeProj\dataset video\test_480_sharp\1.1.png") # same as compare image
compare_image=cv2.imread(r"D:\yt\collegeProj\dataset video\val\val_sharp\000\00000000.png")

test_image=cv2.cvtColor(test_image,cv2.COLOR_BGR2RGB)
transform=transforms.ToTensor()
tensor_test_image=transform(test_image)
tensor_test_image=tensor_test_image.unsqueeze(0)

compare_image=cv2.cvtColor(compare_image,cv2.COLOR_BGR2RGB)
tensor_compare_image=transform(compare_image)
tensor_compare_image=tensor_compare_image.unsqueeze(0)

model=SRCNN()
model.load_state_dict(torch.load(r"SRCNN_1.5X.pth",map_location=device))
model.eval()
criteon=nn.MSELoss()
with torch.no_grad():
    feature=model(tensor_test_image)

loss=criteon(feature,tensor_compare_image)
print("loss:",loss.item())