import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image

class YOLOModel:
    def __init__(self, weights_path : str = "./yolo11l-pose.pt"):
        self.model = YOLO(weights_path)
        self.connections = [(5, 6), (5, 11), (6, 12), (11, 12)]
    
    def predict(self, pil_img : Image.Image):
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=cv_img)
        
        boxes = []
        keypoints_out = []
        for result in results:
            for xyxy, conf in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = xyxy.tolist()
                boxes.append({
                    'x' : float(x1),
                    'y' : float(y1),
                    'width' : float(x2 - x1),
                    'height' : float(y2 - y1),
                    'confidence' : float(conf)
                })
        
            if result.keypoints is not None:
                pts = result.keypoints.xy
                confs = result.keypoints.conf
                for person_xy, person_conf in zip(pts, confs):
                    kp_list = [[float(x), float(y), float(p)] for (x, y), p in zip(person_xy.tolist(), person_conf.tolist())]
                    keypoints_out.append({
                        'keypoints' : kp_list,
                    })
        return boxes, keypoints_out
    
    def draw_annotations(self, pil_img: Image.Image, boxes, keypoints):
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        print(keypoints)
        for x, y, p in keypoints[0]['keypoints']:
            if p > 0.5:
                cv2.circle(cv_img, (int(x), int(y)), 5, (0, 255, 0), -1)
        kp_list = keypoints[0]['keypoints']
        print(kp_list)
        for i, j in self.connections:
            if kp_list[i][2] > 0.5 and kp_list[j][2] > 0.5:
                pt1 = (int(kp_list[i][0]), int(kp_list[i][1]))
                pt2 = (int(kp_list[j][0]), int(kp_list[j][1]))
                cv2.line(cv_img, pt1, pt2, (0, 0, 255), 2)
        
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
