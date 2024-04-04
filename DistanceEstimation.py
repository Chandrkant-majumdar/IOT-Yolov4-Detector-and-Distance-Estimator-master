import cv2 as cv
import numpy as np

# Constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES
VEHICLE_WIDTH = 60.0  # INCHES (Assuming a standard car width)

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONTS = cv.FONT_HERSHEY_COMPLEX

# Load the YOLO model
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Set backend and target to CPU
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Initialize YOLO object detector
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255, swapRB=True)

# Load class names
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Video capture
cap = cv.VideoCapture('demo.mp4')
frame_count = 0  # To skip frames for optimization

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        break

    if frame_count % 2 == 0:  # Skip every other frame for optimization
        continue

    data = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for class_id, confidence, bbox in data:
        class_name = class_names[class_id[0]]
        color = GREEN
        label = f"{class_name}: {confidence}"

        x, y, w, h = bbox
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, label, (x, y - 10), FONTS, 0.5, color, 2)

    # Display the frame
    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
