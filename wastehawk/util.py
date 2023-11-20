import cv2
import numpy as np
import math
import time

# TODO very slow for some reason, need to optimize
# Convert a numpy array or opencv image into something
# that can be used by dearPyGui
def cv2_to_dpg(image, resize_to):
    data = image[:,:,:3] # Remove alpha channel, format is RGB
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) # Correct color
    data = cv2.resize(data, resize_to) # Resize to view frame dimensions
    data = data.ravel()  # Flatten into 1D array
    data = np.asfarray(data, dtype='f')  # Convert data into 32bit floats

    return np.true_divide(data, 255.0)  # Return normalized image data

# Convert dpg 1D array to opencv image. 3 Channels to 4 Channels. 0-1 to 0-255
def dpg_to_cv2(image, target_size):
    data = np.array(image).reshape((target_size[0], target_size[1], 3)) * 255
    data = cv2.cvtColor(data, cv2.COLOR_RGB2RGBA)

    return data

def split_xy(array):
    return [float(x[0]) for x in array], [float(y[1]) for y in array]

# TODO naive implementation, need to optimize
def denoise_points(points: list, resolution = 0.1):
    max_dist = resolution * resolution
    i_p = points.copy()
    c_x = 0
    c_y = 0
    results = []

    count = 0
    t = time.time()

    while len(i_p) != 0:
        point = i_p.pop(0)
        c_x = point[0]
        c_y = point[1]

        for other_point in i_p:
            dist = abs(other_point[0] - c_x) * abs(other_point[1] - c_y)
            if dist <= max_dist:
                c_x = (c_x + other_point[0]) / 2
                c_y = (c_y + other_point[1]) / 2

                i_p.remove(other_point)
                count += 1
            
        results.append([c_x, c_y])
    
    # print(f'Denoise: {time.time() - t:.5f}s, removed {count} items')
    return results


# Calculate offset estimation of object in image.
def estimate_position(points, h_fov, v_fov, shape, drone_x, drone_y, altitude, rotation):
    real_width = (math.tan(math.radians(h_fov)) * altitude) * 2
    real_height = (math.tan(math.radians(v_fov)) * altitude) * 2

    results = []

    for point in points:
        # Project to real world coordinates
        # position offset from center of plane, measured in meters
        x1 = ((point[0] / shape[1]) - 0.5) * real_width
        x2 = ((point[2] / shape[1]) - 0.5) * real_width
        y1 = ((point[1] / shape[0])- 0.5) * real_height
        y2 = ((point[3] / shape[0])- 0.5) * real_height

        # Middle Points
        mx = (x1 + x2) / 2
        my = ((y1 + y2) / 2) * -1

        # Rotation correction
        correction_rot = 0
        if rotation > 180:
            correction_rot = 360 - rotation
        elif rotation <= 180:
            correction_rot = 0 - rotation

        c = math.cos(math.radians(correction_rot))
        s = math.sin(math.radians(correction_rot))

        c_mx = mx * c - my * s
        c_my = mx * s + my * c

        mx = c_mx + drone_x # Apply offsets to drone position
        my = c_my + drone_y

        results.append([mx, my, point[4]])
    
    return results



