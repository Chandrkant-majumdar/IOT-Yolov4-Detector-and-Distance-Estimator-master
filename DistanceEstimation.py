import cv2 as cv
import numpy as np

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES
VEHICLE_WIDTH = 60.0  # INCHES (Assuming a standard car width)

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Load the YOLO model
net = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Set backend and target to CPU for Raspberry Pi
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Initialize YOLO object detector
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255, swapRB=True)

# Object detector function/method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # Creating an empty list to add object's data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # Define color of each object based on its class id
        color = GREEN

        class_id = int(classid)
        label = "%s : %f" % (class_names[class_id], score)

        # Draw rectangle and label on the object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # Getting the data
        # 1: class name, 2: object width in pixels, 3: position where text has to be drawn (distance)
        data_list.append([class_names[class_id], box[2], (box[0], box[1] - 2)])

    # Return list containing the object data
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length


def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance


# Reading the reference images from directory
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels: {person_width_in_rf}, Mobile width in pixels: {mobile_width_in_rf}")

# Finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

# Video capture from file
cap = cv.VideoCapture('demo.mp4')
# cap = cv.VideoCapture(0)
frame_count = 0  # To skip frames for optimization

while True:
    ret, frame = cap.read()
    frame_count += 1

    if not ret:
        break

    if frame_count % 2 == 0:  # Skip every other frame for optimization
        continue

    data = object_detector(frame)
    for d in data:
        if d[0] in ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']:
            if d[0] == 'person':
                distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            elif d[0] == 'car' or d[0] == 'bus' or d[0] == 'truck':
                distance = distance_finder(focal_person, VEHICLE_WIDTH, d[1])
            else:
                # For other classes, assuming the same width as the mobile
                distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    # Display the frame
    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
