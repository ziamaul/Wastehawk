# Wastehawk 0.1

Wastehawk version 0.1, with example GUI application in main.py to work with the drone simulator app used in testing.

Implementation utilities from the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7/tree/main).

## How to use

The demo application takes a video input from a virtual camera, and drone data from localhost port 5005 using a websocket to communicate between applications.

This demo is intended for use with the drone simulator application, which provides an ideal environment for testing the logic of Wastehawk and the demo application.

1. Launch the drone simulator application.
2. Launch OBS Studio, create a new capture in the sources tab to capture footage from the drone simulator application.
3. Start OBS' virtual camera. It should be at device index 1 if your device has an integrated camera.
4. Launch the Wastehawk demo application.

The demo application will work in the background, settings can be adjusted from the side panel in the application.

