from torchvision import transforms
import torch.nn as nn
import torch,cv2,os
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from edge_expert import edge_experts
from texture_expert import texture_experts
from gradient_expert import gradient_expert
from contrast_expert import contrast_experts
from noise_expert import noise_experts
import matplotlib.pyplot as plt

folder_1080_path=r"D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp"
folder_480_path=r"D:\yt\collegeProj\dataset video\train_480_sharp"

folder_1080=sorted(os.listdir(folder_1080_path))
folder_480=sorted(os.listdir(folder_480_path))
file_pair={}

for i in range(len(folder_480)):
    file_pair[folder_1080[i]]=folder_480[i]

print(file_pair)
trans=transforms.ToTensor()
device="cuda" if torch.cuda.is_available() else "cpu"

# layer 1 features
sobel_x,sobel_y,diagonal_left,diagonal_right=edge_experts()
gradient_x,gradient_y,gradient_diag1,gradient_diag2=gradient_expert()

# layer 2 feature
texture_high,texture_lap=texture_experts()

#layer 3 feature
contrast_horizontal,contrast_vertical,contrast_diagonal_2=contrast_experts()

# layer 1 parrallel
noise_high,noise_fine_horizontal,noise_digonal_horizontal=noise_experts()

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

def edge_loss(predict_hr,true_hr):
    predicted_hr=predict_hr.mean(dim=1,keepdim=True)
    true_hr_gray=true_hr.mean(dim=1,keepdim=True)

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

class APEX_SR(nn.Module):
    def __init__(self):
        super().__init__()
        # layer 1 sobels + gradient
        self.sobel_x=nn.Parameter(torch.tensor(sobel_x,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.sobel_y=nn.Parameter(torch.tensor(sobel_y,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.sobel_dig_l=nn.Parameter(torch.tensor(diagonal_left,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.sobel_dig_r=nn.Parameter(torch.tensor(diagonal_right,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)

        self.grad_x=nn.Parameter(torch.tensor(gradient_x,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.grad_y=nn.Parameter(torch.tensor(gradient_y,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.grad_dig_1=nn.Parameter(torch.tensor(gradient_diag1,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.grad_dig_2=nn.Parameter(torch.tensor(gradient_diag2,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.relu1=nn.ReLU(inplace=True)

        # layer1 to layer2 convertion
        self.layer1_to_layer2=nn.Conv2d(in_channels=8,out_channels=3,kernel_size=3,padding=1)

        # layer 2 texture
        self.texture_high=nn.Parameter(torch.tensor(texture_high,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.texture_lap=nn.Parameter(torch.tensor(texture_lap,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.relu2=nn.ReLU(inplace=True)

        # layer 2 to layer 3 convertion
        self.layer2_to_layer3=nn.Conv2d(in_channels=2,out_channels=3,kernel_size=3,padding=1)

        # layer 3 feature contrast
        self.contrast_hori=nn.Parameter(torch.tensor(contrast_horizontal,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.contrast_verti=nn.Parameter(torch.tensor(contrast_vertical,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.contrast_dig_2=nn.Parameter(torch.tensor(contrast_diagonal_2,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.relu3=nn.ReLU(inplace=True)

        #  layer 3 to upscale
        self.layer3_to_upscale=nn.Conv2d(in_channels=3,out_channels=13,kernel_size=3,padding=1)

        # layer parallel feature noise
        self.noise_high=nn.Parameter(torch.tensor(noise_high,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.noise_fine_horizontal=nn.Parameter(torch.tensor(noise_fine_horizontal,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)
        self.noise_dig_hori=nn.Parameter(torch.tensor(noise_digonal_horizontal,dtype=torch.float32).view(1,1,3,3).repeat(1, 3, 1, 1),requires_grad=True)

        # layer upscaling
        self.upscale=nn.Upsample(size=(720,1280),mode="bilinear",align_corners=False)

        # layer convert to 3 outchannel
        self.three_channel=nn.Conv2d(in_channels=8,out_channels=3,kernel_size=3,padding=1,stride=1)

        # enhancement layer
        self.enhancement = nn.Conv2d(16, 32, 3, padding=1)
        self.relu4=nn.ReLU(inplace=True)
        self.enhancement_1=nn.Conv2d(32,16,3,padding=1)
        self.enhace_to_out=nn.Conv2d(16,3,3,padding=1)

    def forward(self,x):
        # layer 1 process sobel + gradient
        layer_edge_x=F.conv2d(x,self.sobel_x,padding=1,stride=1)
        layer_edge_y=F.conv2d(x,self.sobel_y,padding=1,stride=1)
        layer_edge_dig_1=F.conv2d(x,self.sobel_dig_l,padding=1,stride=1)
        layer_edge_dig_2=F.conv2d(x,self.sobel_dig_r,padding=1,stride=1)
        layer_grad_x=F.conv2d(x,self.grad_x,padding=1,stride=1)
        layer_grad_y=F.conv2d(x,self.grad_y,padding=1,stride=1)
        layer_grad_dig_1=F.conv2d(x,self.grad_dig_1,padding=1,stride=1)
        layer_grad_dig_2=F.conv2d(x,self.grad_dig_2,padding=1,stride=1)
        feature_layer_1=torch.cat([layer_edge_x,layer_edge_y,layer_edge_dig_1,layer_edge_dig_2,layer_grad_x,layer_grad_y,layer_grad_dig_1,layer_grad_dig_2],dim=1)

        layer2_input=self.layer1_to_layer2(feature_layer_1)

        # layer 2 process texture
        layer_texture_high=F.conv2d(layer2_input,self.texture_high,padding=1,stride=1)
        layer_texture_lap=F.conv2d(layer2_input,self.texture_lap,padding=1,stride=1)
        feature_layer_2=torch.cat([layer_texture_high,layer_texture_lap],dim=1)
        layer3_input=self.layer2_to_layer3(feature_layer_2)

        # layer 3 contrast
        layer_cont_hori=F.conv2d(layer3_input,self.contrast_hori,padding=1,stride=1)
        layer_cont_verti=F.conv2d(layer3_input,self.contrast_verti,padding=1,stride=1)
        layer_cont_dig2=F.conv2d(layer3_input,self.contrast_dig_2,padding=1,stride=1)
        feature_layer_3=torch.cat([layer_cont_hori,layer_cont_verti,layer_cont_dig2],dim=1)
        layer_cont_input=self.layer3_to_upscale(feature_layer_3)
        layer_cont_cat_input=self.relu3(layer_cont_input)

        # layer parallel noise
        layer_noise_high=F.conv2d(x,self.noise_high,padding=1)
        layer_noise_hori=F.conv2d(x,self.noise_fine_horizontal,padding=1)
        layer_noise_dig1=F.conv2d(x,self.noise_dig_hori,padding=1)
        feature_noise=torch.cat([layer_noise_high,layer_noise_hori,layer_noise_dig1],dim=1)
        layer_noise_cat_input=self.relu2(feature_noise)

        # concat 
        feature_16=torch.cat([layer_cont_cat_input,layer_noise_cat_input],dim=1)

        base=self.upscale(x)
        up=self.upscale(feature_16)
        enhace=self.enhancement(up)
        enhace=self.relu4(enhace)
        enhace=self.enhancement_1(enhace)
        residual=self.enhace_to_out(enhace)
        out_put=base+residual

        return out_put

class ImageLoader(Dataset):
    def __init__(self,path_1080_file,path_1080,path_480_file,path_480):
        self.path_1080=os.path.join(path_1080,path_1080_file) # D:\yt\collegeProj\dataset video\train_sharp\train\train_sharp\000
        self.path_480=os.path.join(path_480,path_480_file) # D:\yt\collegeProj\dataset video\train_480_sharp\000
        self.lr_480_all=sorted(os.listdir(self.path_480)) # D:\yt\collegeProj\dataset video\train_480_sharp\000\00.png
        self.hr_1080_all=sorted(os.listdir(self.path_1080))

        self.transformer=transforms.ToTensor()

    def __len__(self):
        return (len(self.lr_480_all))

    def __getitem__(self, index):
        lr_480_image=cv2.imread(os.path.join(self.path_480,self.lr_480_all[index]))
        lr_480_image=cv2.cvtColor(lr_480_image,cv2.COLOR_BGR2RGB)
        hr_1080_image=cv2.imread(os.path.join(self.path_1080,self.hr_1080_all[index]))
        hr_1080_image=cv2.cvtColor(hr_1080_image,cv2.COLOR_BGR2RGB)

        lr_480_tensor=self.transformer(lr_480_image)
        # hr_1080_c=transforms.CenterCrop((1280,720))
        # hr_1080_cropped=hr_1080_c(hr_1080_image)
        hr_1080_tensor=self.transformer(hr_1080_image)

        return lr_480_tensor,hr_1080_tensor


model=APEX_SR().to(device)
crierion=nn.MSELoss()
optim=torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
edge_weight=0.1
epchos=5
avg_epchho_loss=[]
avg_file_loss=[]
total_file=0
for i in range(epchos):
    total_loss_epcho=0
    average_loss_epcho=0
    total_mse_loss=0
    total_sobel_loss=0
    for key,value in file_pair.items():
        total_loss_file=0
        average_loss_file=0
        sobel_loss=0
        mse_loss=0
        data=ImageLoader(key,folder_1080_path,value,folder_480_path)
        dataset=DataLoader(data,batch_size=4,shuffle=True,num_workers=0)
        current_file=0
        for lr_480,hr_1080 in dataset:
            lr_480=lr_480.to(device,non_blocking=True)
            hr_1080=hr_1080.to(device,non_blocking=True)
            predicted_hr=model(lr_480)

            edge_loss_value=edge_loss(predicted_hr,hr_1080)
            total_sobel_loss+=edge_loss_value.item()
            sobel_loss+=edge_loss_value.item()
            loss=crierion(predicted_hr,hr_1080)
            total_mse_loss+=loss.item()
            mse_loss+=loss.item()
            total_loss=loss+(edge_weight*edge_loss_value)
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            current_file+=1

            total_loss_file+=total_loss.item()
            total_loss_epcho+=total_loss.item()

        average_loss_file=total_loss_file/current_file
        average_mse_loss=mse_loss/current_file
        average_sobel_loss=sobel_loss/current_file

        avg_file_loss.append(average_loss_file)
        total_file+=1
        print("PROCESSED FILE:",key,"Epcho:",i+1,"average mse loss:",average_mse_loss,"average sobel loss:",average_sobel_loss)
    print("Epcho:",i+1,"average epcho mse loss:",(total_mse_loss/(len(file_pair)*len(dataset))),"average epchos sobel loss:",(total_sobel_loss/(len(file_pair)*len(dataset))))
    avg_epchho_loss.append(total_loss_epcho/(11*len(dataset)))

x=avg_file_loss
# print(len(x))
y=list(range(1,total_file+1))
x,y=y,x

plt.plot(x,y)
plt.xlabel("loss of folder")
plt.ylabel("No of folder")
plt.show()


torch.save(model.state_dict(),"APEX_SRCNN_V2.1.pth")