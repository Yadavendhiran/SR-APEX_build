import cv2
from torch.utils.data import DataLoader
import os
import numpy as np

image_path_1080=r"D:\yt\collegeProj\dataset video\val\val_sharp\010"
image_path_480=r"D:\yt\collegeProj\dataset video\val_480_sharp\010"

all_images=sorted(os.listdir(image_path_1080))

if not (os.path.exists(image_path_480)):
    os.makedirs(image_path_480)

    for file in all_images:
        input_image=os.path.join(image_path_1080,file)
        out_image=os.path.join(image_path_480,file)
        og_image=cv2.imread(input_image)
        resize=cv2.resize(og_image,(854,480),interpolation=cv2.INTER_AREA)

        noise=np.random.normal(0,5,resize.shape)
        noisy=resize.astype(np.float32)+noise
        noisy=np.clip(noisy,0,255).astype(np.uint8)

        cv2.imwrite(out_image,noisy)
        print("PROCESSED IMAGE:",input_image)

    print("DONE")
else:
    print("File exists")