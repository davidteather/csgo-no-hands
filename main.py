import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import keyboard
import face_recognition
import math
import ctypes
import functools
import inspect
import time
import json
import base64
from pynput.mouse import Button, Controller
import numpy as np
from unified_detector import Fingertips
import requests
from hand_detector.detector import SOLO, YOLO
import time

hand_detection_method = 'solo'

if hand_detection_method is 'solo':
    hand = SOLO(weights='weights/solo.h5', threshold=0.8)
elif hand_detection_method is 'yolo':
    hand = YOLO(weights='weights/yolo.h5', threshold=0.8)
else:
    assert False, "'" + hand_detection_method + "' hand detection does not exist. use either 'solo' or 'yolo' as hand detection method"

fingertips = Fingertips(weights='weights/classes8.h5')

mouse = Controller()

with open("img_name_to_gun.json", 'r') as i:
    name_to_gun = json.loads(i.read())

with open("csgo-weapon-loadout-hotkeys.json", 'r') as i:
    weapon_hotkeys = json.loads(i.read())

face_default_height = 160
face_default_threshold = 20

mouse_threshold = 40
vert_mouse_threshold = 30

mouse_horizontal_sensitivity = 3.5
mouse_vertical_sensitivity = 1

smooth_movement_split = 2

hand_threshold = 1.0

left_pupil = []
right_pupil = []

# Control Things
is_firing = False
is_w = False
is_s = False

known_face_encodings = []
known_face_names = []

faces = glob.glob("faces_for_loadouts/*")
for face in faces:
    tmp_image = face_recognition.load_image_file(face)
    face_encoding = face_recognition.face_encodings(tmp_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(face.split("\\")[1].split(".")[0])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

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

def face_mouse_movements_old(px, py, pw, ph):
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

def eye_mouse_movements():
    global mouse_threshold
    global vert_mouse_threshold
    screen_width = 1920
    screen_height = 1080

    mid = (math.floor(screen_width/2), math.floor(screen_height/2))
    j = requests.get("http://127.0.0.1:5005/eye").json()

    x_mult = 1
    y_mult = 1

    try:
        if j['x'] < mid[0]:
            x_mult = -1

        if j['y'] < mid[1]:
            y_mult = -1
        
        move_amount = (abs(j['x']-mid[0]), abs(j['y']-mid[1]))

        if move_amount[0] < mouse_threshold:
            x_mult = 0
        
        if move_amount[1] < vert_mouse_threshold:
            y_mult = 0

        move_steps = 20
        for x in range (move_steps):
            move_mouse(math.floor((move_amount[0]*x_mult)/move_steps), math.floor((move_amount[1]*y_mult)/move_steps)) 
    except:
        print("failed probably because no prediction")

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

#video_capture = cv2.VideoCapture(0)

#video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

anterior = 0


while True:
    #if not video_capture.isOpened():
    #    print('Unable to load camera.')
    #    sleep(5)
    #    pass

    # Capture frame-by-frame
    r =  requests.get("http://127.0.0.1:5005/frame")    
    # frame = cv2.UMat(r.content, cv2.COLOR_RGB2GRAY)
    nparr = np.frombuffer(r.content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 

    # cv2.circle(frame, frame_center, radius=2, color=(0, 0, 255), thickness=-1)

    #
    # Facial Recognition
    #
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                loadout_keys = name_to_gun[name].split("&&")
                for weapon in loadout_keys:
                    buttons = weapon_hotkeys[weapon].split("-")
                    keyboard.press_and_release('q')
                    for b in buttons:
                        keyboard.press_and_release(b)

                    for x in range(2):
                        keyboard.press_and_release('escape')
                    

    
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
        # face_mouse_movements(x, y, w, h)
        eye_mouse_movements()
        break

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)


    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    # Display the resulting frame
    # cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
