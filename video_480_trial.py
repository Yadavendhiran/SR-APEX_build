import cv2
import check_1080_480_srcnn
import torch
import numpy
import time
import torch.nn as nn
from srcnn_480_720 import SRCNN

video=cv2.VideoCapture(r"test_480.mp4")
video_ref=cv2.VideoCapture(r"test_720.mp4")
device=check_1080_480_srcnn.device
initial_total_time=time.perf_counter() # total time
model=SRCNN()
model.load_state_dict(torch.load("SRCNN_1.5X_v0.pth",map_location="cpu"))    
model.eval()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
loss=nn.MSELoss()
fps=video.get(cv2.CAP_PROP_FPS)
video_writer=cv2.VideoWriter("output_720_1.mp4",fps=fps,frameSize=(1280,720),fourcc=fourcc)
model_time=0
tranform_time=0
write_time=0
frame_count=0
total_loss=0
while True:
    success,frame=video.read()
    sucess_720,frame_720=video_ref.read()
    if not success and not sucess_720:
            break
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  # CV capture frame -> Tensor
    tensor_frame=check_1080_480_srcnn.transform(frame)
    tensor_frame=tensor_frame.unsqueeze(0)
    frame_count+=1

    test_tensor_frame=check_1080_480_srcnn.transform(frame_720) # 720 -> Tensor
    test_tensor_frame=test_tensor_frame.unsqueeze(0)

    intital_model_time=time.perf_counter() # CV tensor to upscale model
    upscale=model(tensor_frame)
    final_model_time=time.perf_counter()
    model_time+=(final_model_time-intital_model_time)

    ls=loss(upscale,test_tensor_frame)
    total_loss+=ls.item()

    initial_transform_time=time.perf_counter() # Upscaled tensor -> image
    upscale=upscale.squeeze(0)
    upscale=upscale.permute(1,2,0)
    upscale=upscale.detach().numpy()
    upscale=(upscale*255).clip(0,255).astype(numpy.uint8)
    upscale=cv2.cvtColor(upscale,cv2.COLOR_RGB2BGR)
    final_transform_time=time.perf_counter()
    tranform_time+=(final_transform_time-initial_transform_time)

    initial_video_time=time.perf_counter() # iamge/frame -> video *mp4v
    video_writer.write(upscale)
    final_video_time=time.perf_counter()
    write_time+=final_video_time-initial_video_time

video_writer.release()
video.release()
final_total_time=time.perf_counter()
print("FInal time:",((final_total_time-initial_total_time)))
print("total model Time:",model_time)
print("total transform Time:",tranform_time)
print("total write time:",write_time)
print("frame:",frame_count)
print("average loss:",(total_loss/frame_count))