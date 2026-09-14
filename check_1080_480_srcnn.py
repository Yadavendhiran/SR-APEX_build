import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from srcnn_480_720 import SRCNN

device="cuda"


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
model.load_state_dict(torch.load(r"SRCNN_1.5X_v0.pth",map_location=device))
model.eval()
criteon=nn.MSELoss()
with torch.no_grad():
    feature=model(tensor_test_image)

loss=criteon(feature,tensor_compare_image)
print("loss:",loss.item())