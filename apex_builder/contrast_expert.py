# from torchvision import transforms
# import torch.nn as nn
# import torch,cv2,os
# import torch.nn.functional as F

# # hypotesis use tensor 1,2,5

# lr_img_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\001"
# hr_image_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000"

# lr_image_all=sorted(os.listdir(lr_img_path))
# hr_image_all=sorted(os.listdir(hr_image_path))

# trans=transforms.ToTensor()
# device="cuda" if torch.cuda.is_available() else "cpu"

# contrast_horizontal=torch.tensor(
#     [
#         [-1,-1,-1],
#         [2,2,2],
#         [-1,-1,-1]
#     ],dtype=torch.float32
# ).view(1,1,3,3)

# contrast_vertical=torch.tensor(
#     [
#         [-1,2,-1],
#         [-1,2,-1],
#         [-1,2,-1]
#     ],dtype=torch.float32
# ).view(1,1,3,3)

# # contrast_diagonal_1=torch.tensor(
# #     [
# #         [2,-1,-1],
# #         [-1,2,-1],
# #         [-1,-1,2]
# #     ],dtype=torch.float32
# # ).view(1,1,3,3)

# contrast_diagonal_2=torch.tensor(
#     [
#         [-1,-1,2],
#         [-1,2,-1],
#         [-2,-1,-1]
#     ],dtype=torch.float32
# ).view(1,1,3,3)

# # experts=[contrast_horizontal,contrast_vertical,contrast_diagonal_1,contrast_diagonal_2]

# lr_tensors=[]
# hr_tensors=[]
# for i in range(len(hr_image_all)):
#     hr_img=cv2.imread(os.path.join(hr_image_path,hr_image_all[i]))
#     lr_img=cv2.imread(os.path.join(lr_img_path,lr_image_all[i]))
    
#     lr_image=cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
#     lr_tensor=trans(lr_image).unsqueeze(0)
    
#     hr_image=cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
#     # hr_image=cv2.resize(hr_image,(854,480),interpolation=cv2.INTER_AREA)
#     hr_tensor=trans(hr_image).unsqueeze(0)

#     lr_tensors.append(lr_tensor.mean(dim=1,keepdim=True))
#     hr_tensors.append(hr_tensor.mean(dim=1,keepdim=True))




# # lr_obvs=[]
# # hr_obvs=[]
# # for i in experts:
# #     lr_result=[]
# #     hr_result=[]
# #     for lr,hr in zip(lr_tensors,hr_tensors):
# #         lr_r=F.conv2d(lr.to(device,non_blocking=True),i.to(device,non_blocking=True),padding=1)
# #         hr_r=F.conv2d(hr.to(device,non_blocking=True),i.to(device,non_blocking=True),padding=1)
# #         lr_result.append(lr_r.detach().cpu().numpy())
# #         hr_result.append(hr_r.detach().cpu().numpy())   
# #     lr_obvs.append(lr_result)
# #     hr_obvs.append(hr_result)


# # sum_lr=[]
# # for i in range(len(lr_obvs)):
# #     sum_lr_exp=0
# #     for j in range(len(lr_obvs[i])):
# #         sum_lr_exp+=lr_obvs[i][j].sum()
# #     sum_lr.append(sum_lr_exp)
# # print("sum of lr :",sum_lr)

# # sum_hr=[]
# # for i in range(len(hr_obvs)):
# #     sum_hr_exp=0
# #     for j in range(len(hr_obvs[i])):
# #         sum_hr_exp+=hr_obvs[i][j].sum()
# #     sum_hr.append(sum_hr_exp)
# # print("sum of hr :",sum_hr)

def contrast_experts():
    contrast_horizontal=[
            [-1,-1,-1],
            [2,2,2],
            [-1,-1,-1]
        ]

    contrast_vertical=[
            [-1,2,-1],
            [-1,2,-1],
            [-1,2,-1]
        ]

    contrast_diagonal_2=[
            [-1,-1,2],
            [-1,2,-1],
            [-2,-1,-1]
        ]

    return contrast_horizontal,contrast_vertical,contrast_diagonal_2