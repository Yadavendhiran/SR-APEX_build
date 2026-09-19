from sobel_loss import edge_loss
import cv2,torch,numpy
from torchvision import transforms
from srcnn_480_720 import SRCNN
from time import perf_counter

device="cpu"

model_load_time_start=perf_counter()
model=SRCNN().to(device)
model.load_state_dict(
    torch.load(r"D:\yt\collegeProj\model_480_to_720\SRCNN_1.5X_v1.3.pth")
)
model.eval()
model_load_time_end=perf_counter()
transform=transforms.ToTensor()
print("model loaded time taken :",(model_load_time_end-model_load_time_start))

video_path=r"D:\yt\collegeProj\test_480_720\test_480.mp4"
video_read=cv2.VideoCapture(video_path)
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
fps=video_read.get(cv2.CAP_PROP_FPS)
video_write=cv2.VideoWriter("apex_output.mp4",fourcc=fourcc,frameSize=(1280,720),fps=fps)

check_state=0 # check only from second frame skip the first frame

model_time=0
similar_frame=0
diff_frame=0
while True:
    success,frame=video_read.read()
    if not success:
        break

    if check_state==0:
        current_frame=frame
        frame_pre_process=cv2.cvtColor(current_frame,cv2.COLOR_BGR2RGB)
        frame_tensor=transform(frame_pre_process)
        frame_tensor=frame_tensor.unsqueeze(0)

        model_start_time=perf_counter()
        upscale_feature=model(frame_tensor)
        model_end_time=perf_counter()
        model_time+=(model_end_time-model_start_time)

        tensor_to_frame=upscale_feature.squeeze(0)
        tensor_to_frame=tensor_to_frame.permute(1,2,0)
        tensor_to_frame=tensor_to_frame.detach().numpy()
        tensor_to_frame=(tensor_to_frame*255).clip(0,255).astype(numpy.uint8)
        tensor_to_frame=cv2.cvtColor(tensor_to_frame,cv2.COLOR_RGB2BGR)

        video_write.write(tensor_to_frame)
        global temp_upscale_memory
        temp_upscale_memory=upscale_feature # contains upscale memory temp use
        check_state+=1
    else:
        next_frame=frame
        tensor_next_frame=cv2.cvtColor(next_frame,cv2.COLOR_BGR2RGB)
        tensor_next_frame=transform(tensor_next_frame)
        tensor_next_frame=tensor_next_frame.unsqueeze(0)

        tensor_current_frame=cv2.cvtColor(current_frame,cv2.COLOR_BGR2RGB)
        tensor_current_frame=transform(tensor_current_frame)
        tensor_current_frame=tensor_current_frame.unsqueeze(0)

        sobel_loss=edge_loss(tensor_next_frame,tensor_current_frame)
        if(sobel_loss.item()>=0.15):
            diff_frame+=1
            model_start_time=perf_counter()
            upscale_feat=model(tensor_next_frame)
            model_end_time=perf_counter()
            model_time+=(model_end_time-model_start_time)
            upscale=upscale_feat.squeeze(0)
            upscale=upscale.permute(1,2,0)
            upscale=upscale.detach().numpy()
            upscale=(upscale*255).clip(0,255).astype(numpy.uint8)
            upscale=cv2.cvtColor(upscale,cv2.COLOR_RGB2BGR)

            video_write.write(upscale)
            current_frame=next_frame
            temp_upscale_memory=upscale_feat
        else:
            similar_frame+=1
            upscale_feat=temp_upscale_memory
            upscale=upscale_feat.squeeze(0)
            upscale=upscale.permute(1,2,0)
            upscale=upscale.detach().numpy()
            upscale=(upscale*255).clip(0,255).astype(numpy.uint8)
            upscale=cv2.cvtColor(upscale,cv2.COLOR_RGB2BGR)

            video_write.write(upscale)


video_write.release()
video_read.release()
print("model time: ",model_time)
print("similar frame: ",similar_frame,"diffrent frame :",diff_frame)