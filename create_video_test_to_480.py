import cv2

video=cv2.VideoCapture(r"A:\like mona lisa.mp4")
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
fps=video.get(cv2.CAP_PROP_FPS)
video_writer_480=cv2.VideoWriter("test_480.mp4",fourcc=fourcc,fps=fps,frameSize=(854,480))
video_writer_720=cv2.VideoWriter("test_720.mp4",fourcc=fourcc,fps=fps,frameSize=(1280,720))

while True:
    success,frame=video.read()
    if not success:
        break
    reduce_480=cv2.resize(frame,(854,480),interpolation=cv2.INTER_AREA)
    reduce_720=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)

    video_writer_480.write(reduce_480)
    video_writer_720.write(reduce_720)

video_writer_720.release()
video_writer_480.release()
video.release()
