import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import keyboard
import math
import ctypes
import functools
import inspect
import time
from pynput.mouse import Button, Controller
import numpy as np
from unified_detector import Fingertips
from hand_detector.detector import SOLO, YOLO

hand_detection_method = 'solo'

if hand_detection_method is 'solo':
    hand = SOLO(weights='weights/solo.h5', threshold=0.8)
elif hand_detection_method is 'yolo':
    hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
else:
    assert False, "'" + hand_detection_method + "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"

fingertips = Fingertips(weights='weights/classes8.h5')

mouse = Controller()


face_default_height = 160
face_default_threshold = 20

mouse_threshold = 45
vert_mouse_threshold = 40

mouse_horizontal_sensitivity = 3.5
mouse_vertical_sensitivity = 1

smooth_movement_split = 2

hand_threshold = 1.0

# Control Things
is_firing = False
is_w = False
is_s = False

def move_mouse(x, y):
    ctypes.windll.user32.mouse_event(
        ctypes.c_uint(0x0001),
        ctypes.c_uint(math.floor(x)),
        ctypes.c_uint(math.floor(y)),
        ctypes.c_uint(0),
        ctypes.c_uint(0)
    )

def face_size_parser(x, y, w, h):
    global is_s
    global is_w
    if h > face_default_height+face_default_threshold:
        print("forward")
        #if not is_w:
        #    keyboard.press("w")
        #    is_w = True
        keyboard.press("w")
    elif h < face_default_height-(face_default_threshold*1.2):
        print("backward")
        #if not is_s:
        #    keyboard.press("s")
        #    is_s = True
        keyboard.press("s")
    else:
        print("in threshold")
        keyboard.release("w")
        keyboard.release("s")
        is_w = False
        is_s = False

def check_if_shoot(image):
    global is_firing
    is_shooting = False
    tl, br = hand.detect(image=image)
    if tl and br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        # gesture classification and fingertips regression
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)

        # post-processing
        prob = np.asarray([(p >= hand_threshold) * 1.0 for p in prob])
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        # drawing
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
        for c, p in enumerate(prob):
            if p >= hand_threshold:
                print("hand detected")
                is_shooting = True
                # break
                print(p)
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
            index = index + 2

    if is_shooting:
            print('firing')
            mouse.press(Button.left)
            is_firing = True
    else:
        mouse.release(Button.left)
        is_firing = False

def face_mouse_movements(px, py, pw, ph):
    x = math.floor(pw/2)+px
    y = math.floor(ph/2)+py
    # x axis movement
    if x < frame_center[0] - mouse_threshold:

        for x in range(math.floor(math.floor(abs(((frame_center[0] - mouse_threshold)-x))*(mouse_horizontal_sensitivity))/smooth_movement_split)):
            move_mouse(smooth_movement_split,0)

    elif x > frame_center[0] + mouse_threshold:
        #for x in range(abs(math.floor(math.floor(((frame_center[0] + mouse_threshold)-x)*(mouse_horizontal_sensitivity))/smooth_movement_split))):
        #    move_mouse(-1*smooth_movement_split, 0)
        mov = math.floor(math.floor(((frame_center[0] + mouse_threshold)-x)*(mouse_horizontal_sensitivity))/smooth_movement_split)
        if abs(mov) > 30:
            move_mouse(mov, 0)
        else:
            move_mouse(mov/2, 0)

    # y axis movement
    if y > frame_center[1] + vert_mouse_threshold:
        move_mouse(0,math.floor(abs(((frame_center[1] - vert_mouse_threshold)-y))*(mouse_vertical_sensitivity)))

    elif y < frame_center[1] - vert_mouse_threshold:
        move_mouse(0,math.floor(((frame_center[1] + vert_mouse_threshold)-y)*(-1*mouse_vertical_sensitivity)))

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)

#video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

anterior = 0

ret, frame = video_capture.read()
f_h, f_w, channels = frame.shape
frame_center = (math.floor(f_w/2), math.floor(f_h/2))

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    cv2.circle(frame, frame_center, radius=2, color=(0, 0, 255), thickness=-1)
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # detect shoot
    check_if_shoot(frame)

    # Draw a rectangle around the faces
    # modified to only show one face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_size_parser(x, y, w, h)
        face_mouse_movements(x, y, w, h)
        break

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
