# Real-time Object Detection using YOLOv5 and Ultralytics

This Python script demonstrates real-time object detection using the YOLOv5 model from Ultralytics. It captures video from the default camera, detects objects in the video frames, and draws bounding boxes around the detected objects.

![Picture1](https://github.com/VibhorX/ObjectDifferentiation/assets/110552245/8f06d115-19ef-4d5a-b141-491f91e7500d)


## Requirements

- Python 3.x
- OpenCV
- YOLOv5 (from Ultralytics)

## Usage

1. Make sure you have all the required libraries installed (`pip install opencv-python`).
2. Download the YOLOv5 model weights from the Ultralytics repository.
3. Run the Python script (`python object_detection.py`).
4. Position yourself in front of the camera.
5. The script will detect and draw bounding boxes around objects in real-time.

## Description

- The script uses the YOLOv5 model from Ultralytics for object detection.
- It captures video frames from the default camera (you can change the camera resolution if needed).
- The YOLOv5 model detects objects in the video frames.
- Bounding boxes are drawn around the detected objects using OpenCV's `rectangle` function.
- The class name and confidence score of each detected object are displayed on the video feed.

## Additional Notes

- YOLOv5 is a fast and accurate object detection model known for its performance and ease of use.
- You can experiment with different model variants (e.g., YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x) and model weights for different detection tasks.
- This script provides a simple demonstration of real-time object detection and may require optimizations for specific use cases or environments.
