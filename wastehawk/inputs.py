import cv2
import csv
import time
import math
from pygrabber.dshow_graph import FilterGraph

instances = []

class VideoInput:
    def __init__(self, index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        self.cap = cap
        self.frame_shape = (cap.get(3), cap.get(4))
        instances.append(self)
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def read(self):
        if not self.isOpened():
            #print("Capture closed")
            return [None, None]
        
        ret, frame  = self.cap.read()
        
        return [ret, frame]
    
    def close(self):
        self.cap.release()

class SyncedVideoInput:
    def __init__(self, video_path, data_path, data_frequency, start_time = 0):
        self.cap = cv2.VideoCapture(video_path)
        self.offset_time = start_time
        self.start_time = time.time()
        self.data_frequency = data_frequency

        print(self.cap.isOpened())

        instances.append(self)

        with open(data_path) as file:
            reader = csv.reader(file, delimiter=',')
            self.data = list(reader)
    
    def _get_data(self):
        t = ((time.time() + self.offset_time) - self.start_time) * self.data_frequency
        row_index = math.floor(t)
        data = self.data[row_index]
        return data
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def read(self):
        if not self.isOpened():
            print("Capture closed")
            return [None, None]
        
        ret, frame  = self.cap.read()
        
        return [ret, frame]
    
    def close(self):
        self.cap.release()

def get_cameras():
    devices = FilterGraph().get_input_devices()
    
    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras

def close():
    for instance in instances:
        instance.close()