import tkinter as tk
import cv2
from ultralytics import YOLO
import math
import PIL.Image
import PIL.ImageTk

# Initialize the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Define object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Flag to control video feed inversion
invert_video = True

# Function to continuously capture frames from the webcam and perform object detection
def detect_objects_live():
    cap = cv2.VideoCapture(0)  # Use the default camera (change the index if you have multiple cameras)

    while True:
        success, frame = cap.read()

        if not success:
            break

        if invert_video:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        results = model(frame)

        # Process and display object detections
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])

                # Draw the bounding box with a unique color for each class
                color = (0, 255, 0)  # Default to green
                if 0 <= cls < len(classNames):
                    if classNames[cls] == "person":
                        color = (0, 255, 0)  # Green frame for "person"
                    else:
                        color = (0, 0, 255)  # Red frame for other classes

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = PIL.Image.fromarray(frame_rgb)
        frame_tk = PIL.ImageTk.PhotoImage(image=frame_pil)

        # Update the label with the processed frame
        image_label.config(image=frame_tk)
        image_label.image = frame_tk

        root.update()  # Update the GUI window

    cap.release()

# Function to toggle video feed inversion
def toggle_invert():
    global invert_video
    invert_video = not invert_video

# Create the main GUI window
root = tk.Tk()
root.title("Visual Differentiation Project")

# Create a label to display the webcam feed with object detections
image_label = tk.Label(root)
image_label.pack()

# Create a button to start live object detection
start_button = tk.Button(root, text="Start", command=detect_objects_live)
start_button.pack()

# Create a button to inverse video feed
start_button = tk.Button(root, text="Inverse Video", command=toggle_invert)
start_button.pack()


# Start the GUI event loop
root.mainloop()

