from sobel_loss import edge_loss
import torch,cv2
from torchvision import transforms

read_v1_input=cv2.VideoCapture(r"D:\yt\collegeProj\480_720_converter\output_720_14.mp4")  # V1 
read_v2_input=cv2.VideoCapture(r"D:\yt\collegeProj\apex_builder\apex_out_8.mp4")  # V2
read_tru_hr=cv2.VideoCapture(r"E:\test 720 file\test_720.mp4") # hr

transform=transforms.ToTensor()
device="cuda" if torch.cuda.is_available() else "cpu"
loss_in_1=0
loss_in_2=0
frames=0
if int(read_v2_input.get(cv2.CAP_PROP_FRAME_COUNT))==int(read_v2_input.get(cv2.CAP_PROP_FRAME_COUNT)):
    while True:
        
        success,frame=read_v2_input.read()
        s1,f1=read_v1_input.read()
        s_hr,f_hr=read_tru_hr.read()
        if not success and not s1 and not s_hr:
            break
        
        frames+=1
        current_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)    # V2 frames
        current_frame=transform(current_frame).unsqueeze(0).to(device)
        
        current_hr_frame=cv2.cvtColor(f_hr,cv2.COLOR_BGR2RGB)  # HR frames
        current_hr_frame=transform(current_hr_frame).unsqueeze(0).to(device)
        loss_2=edge_loss(current_frame,current_hr_frame)
        loss_in_2+=loss_2.item()
        
        current_1_f=cv2.cvtColor(f1,cv2.COLOR_BGR2RGB)  # V1 frames
        current_1_f=transform(current_1_f).unsqueeze(0).to(device)
        loss_1=edge_loss(current_1_f,current_hr_frame)
        loss_in_1+=loss_1.item()
        
        
print("sobel loss of SR V1:")
print(loss_in_1, frames)

print("sobel loss of SR V2:")
print(loss_in_2, frames)