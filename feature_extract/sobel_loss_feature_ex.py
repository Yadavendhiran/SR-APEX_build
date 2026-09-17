import torch
import torch.nn as nn

device="cpu"

sobelx=torch.tensor(
    [[-1.0,0.0,1.0],
     [-2.0,0.0,2.0],
     [-1.0,0.0,1.0]],dtype=torch.float32
).view(1,1,3,3).to(device)

sobely=torch.tensor(
    [[-1.0,-2.0,-1.0],
     [0.0,0.0,0.0],
     [1.0,2.0,1.0]],dtype=torch.float32
).view(1,1,3,3).to(device)


# edge loss using sobel
def edge_loss(predict_hr,true_hr):
    predicted_hr=predict_hr.mean(dim=1,keepdim=True)
    true_hr_gray=true_hr.mean(dim=1,keepdim=True)
    predicted_hr=nn.functional.interpolate(
            predicted_hr,size=true_hr_gray.shape[-2:],
            mode="bilinear",align_corners=True
        )
    predicted_x=nn.functional.conv2d(
        predicted_hr,sobelx,padding=1
    )
    predicted_y=nn.functional.conv2d(
        predicted_hr,sobely,padding=1
    )

    true_hr_x=nn.functional.conv2d(
        true_hr_gray,sobelx,padding=1
    )
    true_hr_y=nn.functional.conv2d(
        true_hr_gray,sobely,padding=1
    )
    loss_x=nn.functional.l1_loss(predicted_x,true_hr_x)
    loss_y=nn.functional.l1_loss(predicted_y,true_hr_y)
    return loss_x+loss_y
