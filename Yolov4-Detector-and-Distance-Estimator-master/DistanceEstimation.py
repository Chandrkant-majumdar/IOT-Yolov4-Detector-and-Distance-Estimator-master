import cv2 as cv
import numpy as np

# import pygame


# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES
VEHICLE_WIDTH = 60.0  # INCHES (Assuming a standard car width)

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up OpenCV net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)


# Object detector function/method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # Creating an empty list to add object's data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # Define color of each object based on its class id
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)

        # Draw rectangle and label on the object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # Getting the data
        # 1: class name, 2: object width in pixels, 3: position where text has to be drawn (distance)
        data_list.append([class_names[classid], box[2], (box[0], box[1] - 2)])

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
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels: {person_width_in_rf}, Mobile width in pixels: {mobile_width_in_rf}")

# Finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

# Video capture from file
#cap = cv.VideoCapture(0);
cap = cv.VideoCapture(r"C:\Users\chand\Downloads\Yolov4-Detector-and-Distance-Estimator-master\Yolov4-Detector-and-Distance-Estimator-master\demo.mp4")

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output_video.avi', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))



# # Initialize pygame mixer
# pygame.mixer.init()
#
# # Load the alert sound
# alert_sound = pygame.mixer.Sound(r'C:\Users\chand\Downloads\Yolov4-Detector-and-Distance-Estimator-master\Yolov4-Detector-and-Distance-Estimator-master\alert_sound.mp3')
#
# # Threshold distance for triggering the alert (in inches)
# threshold_distance = 24  # Adjust as needed
#
# # Dictionary to store the state of each object and whether sound is playing for it
# object_states = {}
#
#
# # Function to play sound alert
# def play_alert():
#     # Play the alert sound
#     alert_sound.play()
#

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flag to indicate if any object is currently within threshold distance
    # object_detected = False

    data = object_detector(frame)
    for d in data:
        if d[0] in ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter']:
            if d[0] == 'person':
                distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            elif d[0] == 'car':
                distance = distance_finder(focal_person, VEHICLE_WIDTH, d[1])
            else:
                # For other classes, assuming the same width as the mobile
                distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])

            x, y = d[2]

            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

            # Check if distance is less than the threshold for triggering the alert
            # if distance < threshold_distance:
            #     object_detected = True
            #     # Check if the sound is not already playing for this object
            #     if d[0] not in object_states or not object_states[d[0]]:
            #         play_alert()  # Play sound alert
            #         object_states[d[0]] = True
            # else:
            #     # Reset the state for this object if it moves away
            #     object_states[d[0]] = False

    # Stop the sound if no objects are detected or if all objects moved out of threshold distance
    # if not object_detected:
    #     alert_sound.stop()
    #     # Reset the states for all objects
    #     object_states = {}

    # Write the frame into the output video file
    out.write(frame)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()