from ultralytics import YOLO
import os
import cv2
import random
from wrapper import Wrapper

v_path = os.path.join('.', 'people.mp4')
video_out_path = os.path.join('.', 'out.mp4')

if not os.path.exists('images'):
    os.makedirs('images')
    
cap = cv2.VideoCapture(v_path)
ret, frames = cap.read()
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),(frames.shape[1], frames.shape[0]))

model = YOLO("yolov8n.pt")
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(15)]
tracker = Wrapper()

count = 0
threshold = 0.5

while ret:
    result_frame = model(frames)
    for frame in result_frame:
        detections = []
        for r in frame.boxes.data.tolist():
            x1, y1, w, h, score, class_id = r
            x1, y1, w, h, class_id = map(int,(x1, y1, w, h, class_id))
            
            if score > threshold:
                detections.append([x1, y1, w, h, score])
        tracker.update(frames, detections)
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, w, h = bbox
            track_id = track.track_id
            
            cv2.rectangle(frames, (int(x1), int(y1)), (int(w), int(h)), (colors[track_id % len(colors)]), 3)
        count +=1
        if 0 < count < 100:
            cv2.imwrite('./images/' + str(count) +'.png', frames)

    cap_out.write(frames)
    ret, frames = cap.read()
cap.release()
cap_out.release()
cv2.destroyAllWindows()		 