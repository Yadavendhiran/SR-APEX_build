import cv2

video_path=r"A:\like mona lisa final.mp4"

video=cv2.VideoCapture(video_path)
fourcc=cv2.VideoWriter_fourcc(*"mp4v")
fps=video.get(cv2.CAP_PROP_FPS)

video_write=cv2.VideoWriter("test_180.mp4",frameSize=(320,180),fourcc=fourcc,fps=fps)
write_720=cv2.VideoWriter("test_720.mp4",frameSize=(1280,720),fourcc=fourcc,fps=fps)

while True:
    success,frame=video.read()
    if not success:
        break

    og_frame=frame
    frame_180=cv2.resize(og_frame,(320,180),interpolation=cv2.INTER_AREA)
    frame_720=cv2.resize(og_frame,(1280,720),interpolation=cv2.INTER_AREA)
    video_write.write(frame_180)
    write_720.write(frame_720)

video_write.release()
write_720.release()
video.release()