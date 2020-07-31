import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
import ctypes
import math
import pyautogui
import pydirectinput

def move_mouse(x, y):
    ctypes.windll.user32.mouse_event(
        ctypes.c_uint(0x0001),
        ctypes.c_uint(math.floor(x)),
        ctypes.c_uint(math.floor(y)),
        ctypes.c_uint(0),
        ctypes.c_uint(0)
    )

left_pupil = []
right_pupil = []

_, frame = webcam.read()
def set_config():
    global left_pupil
    global right_pupil
    global webcam
    global gaze


    for place in ['top left', 'top right', 'bottom left', 'bottom right']:
        print("Look at the {} of your screen and hit enter".format(place))
        input()

        _, frame = webcam.read()
        gaze.refresh(frame)

        left_pupil.append(gaze.pupil_left_coords())
        right_pupil.append(gaze.pupil_right_coords())

def eye_coords_to_screen(coords, width, height):
    global left_pupil
    global right_pupil

    screenPos = [(0,0), (1920,0), (0,1080), (1920,1080)]



set_config()
while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    new_frame = gaze.annotated_frame()
    text = ""

    if gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(new_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
    cv2.imshow("Demo", new_frame)

    if len(right_pupil) == 4:
        # Then now we do the thing
        screen_width = 1920
        screen_height = 1080

        dist_between_x = left_pupil[0][0] - left_pupil[1][0]
        per_pixel_x = screen_width/dist_between_x

        dist_between_y = left_pupil[0][1] - left_pupil[2][1]
        per_pixel_y = screen_width/dist_between_x

        coords = gaze.pupil_left_coords()
        if coords != None:
            pydirectinput.moveTo(math.floor(abs((coords[0]-left_pupil[0][0])*per_pixel_x)), math.floor(abs((coords[1]-left_pupil[0][1])*per_pixel_y)))


    if cv2.waitKey(1) == 27:
        break