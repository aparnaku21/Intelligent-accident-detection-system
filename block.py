import cv2
# from pred import get_speeds
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation

model = YOLO("yolov8n.pt")
names = model.model.names

def block(path):
    result = {}
    result= {
        'accident':False,
        'criminal_detected':False,
        'trafic_detected':False
    }
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
    video_writer = cv2.VideoWriter("speed_estimation.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

    line_pts = [(0, 360), (1280, 360)]

# Init speed-estimation obj
    speed_obj = speed_estimation.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)

    while cap.isOpened():

        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        tracks = model.track(im0, persist=True, show=False)
    
        im0 = speed_obj.estimate_speed(im0, tracks)
        # print('speed:', tracks[0].person)
        print(speed_obj.dist_data)
        for track_id, speed in speed_obj.dist_data.items():
            if speed < 10:
                print("Traffic block detected")
                result['trafic_detected'] = True
                cap.release()
                break
        video_writer.write(im0)
 
   

   
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    return result 