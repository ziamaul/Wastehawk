import time
import math
import wastehawk.util as util

from multiprocessing.pool import ThreadPool, AsyncResult
from wastehawk.detector import Detector

class Wastehawk:
    def __init__(self):

        self.configs = {
            'h_fov': 0.0,
            'v_fov': 0.0,
            'denoise_resolution': 0.1
        }

        # Drone Positions
        self.drone = {
            'x':0.0,
            'y':0.0,
            'altitude':0.0,
            'heading':0.0,
            'origin': None,
        }
        self.drone_path = []

        # Detection & Threading
        self._detector: Detector = None
        self._threadpool: ThreadPool = ThreadPool(processes=1)
        self._detect_thread: AsyncResult = None

        # Output
        self.trash_positions: list = []
        self.output: list = []
        self.last_frame = None

        # Debug
        self.latency = 0.0 # in seconds
        self.denoise_latency = 0.0
        self._latency_marker = 0.0

    def apply_configs(self, h_fov: float = 0, v_fov: float = 0, denoise_resolution: float = 0.1):
        self.configs['h_fov'] = h_fov
        self.configs['v_fov'] = v_fov
        self.configs['denoise_resolution'] = denoise_resolution

        return self

    def set_detector(self, detector: Detector):
        self._detector = detector

        return self

    # Heading is measured in clockwise, with north being 0 degrees, east 90, south 180, west 270.
    # x, y, altitude measured in meters.
    def update_drone_data(self, x, y, altitude, heading):
        if self.drone['origin'] == None:
            self.drone['origin'] = [x, y]

        self.drone['x'] = x - self.drone['origin'][0]
        self.drone['y'] = y - self.drone['origin'][1]
        self.drone['altitude'] = altitude
        self.drone['heading'] = heading


        self.drone_path.append([x, y, altitude, heading])

    # Snapshot of the drone data at the current time
    def get_drone_data_snapshot(self):
        return [self.drone['x'], self.drone['y'], self.drone['altitude'], self.drone['heading']]
    
    def _process_results(self, shape, results, data):
        # Projection to real-world coordinates
        projected_plots = util.estimate_position(results, self.configs['h_fov'], self.configs['v_fov'], shape, data[0], data[1], data[2], data[3])
        
        
        for plot in projected_plots:
            self.trash_positions.append([
                plot[0],
                plot[1]
            ])
        
        # Denoising data
        t = time.time()
        self.trash_positions = util.denoise_points(self.trash_positions, resolution=self.configs['denoise_resolution'])
        self.denoise_latency = time.time() - t

    def _detect_function(self, detector, frame, data):
        shape, results = detector._detect(frame)
        return shape, results, data
    
    # Updates detection
    def update_detection(self, frame):
        assert self._detector != None

        if not frame.any():
            return
        
        if self._detect_thread != None and self._detect_thread.ready():
            shape, results, data = self._detect_thread.get()
            self._process_results(shape, results, data)
            self._detect_thread = None
            
            self.output = [shape, results, data]
            self.latency = (time.time() - self._latency_marker)
            
            
        if self._detect_thread == None:
            self._latency_marker = time.time()
            self._detect_thread = self._threadpool.apply_async(self._detect_function, (self._detector, frame, self.get_drone_data_snapshot()))