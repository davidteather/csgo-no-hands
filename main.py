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


face_default_height = 160
face_default_threshold = 20

mouse_threshold = 55
vert_mouse_threshold = 50

mouse_horizontal_sensitivity = 2
mouse_vertical_sensitivity = 1.1

def move_mouse(x, y):
    ctypes.windll.user32.mouse_event(
        ctypes.c_uint(0x0001),
        ctypes.c_uint(x),
        ctypes.c_uint(y),
        ctypes.c_uint(0),
        ctypes.c_uint(0)
    )

def face_size_parser(x, y, w, h):
    if h > face_default_height+face_default_threshold:
        print("forward")
        keyboard.press("w")

    elif h < face_default_height-(face_default_threshold*1.2):
        print("backward")
        keyboard.press("s")
    else:
        print("in threshold")
        keyboard.release("w")
        keyboard.release("s")

def face_mouse_movements(px, py, pw, ph):
    x = math.floor(pw/2)+px
    y = math.floor(ph/2)+py
    # x axis movement
    if x < frame_center[0] - mouse_threshold:
        move_mouse(math.floor(abs(((frame_center[0] - mouse_threshold)-x))*(mouse_horizontal_sensitivity)),0)

    elif x > frame_center[0] + mouse_threshold:
        move_mouse(math.floor(((frame_center[0] + mouse_threshold)-x)*(mouse_horizontal_sensitivity)), 0)

    # y axis movement
    if y > frame_center[1] + vert_mouse_threshold:
        move_mouse(0,math.floor(abs(((frame_center[1] - vert_mouse_threshold)-y))*(mouse_vertical_sensitivity)))

    elif y < frame_center[1] - vert_mouse_threshold:
        move_mouse(0,math.floor(((frame_center[1] + vert_mouse_threshold)-y)*(-1*mouse_vertical_sensitivity)))

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
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
