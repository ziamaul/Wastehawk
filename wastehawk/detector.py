import time

import cv2
import torch
import numpy as np

from utils.general import non_max_suppression
from utils.datasets import letterbox

class Detector:
    def __init__(self, _weights='app/resources/weights.pt', _device='cpu'):
        t0 = time.time()

        ### Load model
        self.model = torch.load(_weights, _device)['model'].float().fuse().eval()
        self.device = torch.device(_device)
        
        print(f'Load Model. ({time.time() - t0:.3f}s)')

    # Implementation taken from the YOLOv7 repository: https://github.com/WongKinYiu/yolov7
    def _detect(self, _source: cv2.Mat, conf_thres=0.25, iou_thres=0.45, img_size=640, stride=32):
        t0 = time.time()
        input_image = _source
        
        image = letterbox(input_image, img_size, stride=stride)[0]
        _shape = image.shape
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)

        ### Inference
        img = torch.from_numpy(image).to(self.device).float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        t1 = time.time()
        with torch.no_grad():
            pred = self.model(img)[0]
        

        t2 = time.time()
        results = non_max_suppression(pred, conf_thres, iou_thres)[0] # Only run one detection
        t3 = time.time()

        print(f'Detect. ({time.time() - t0:.3f}s: Prep {t1-t0:.5f}s, Inference {t2-t1:.5f}s, NMS {t3-t2:.5f}s)')

        return [_shape, results]

