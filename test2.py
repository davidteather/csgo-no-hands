from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from gaze_codefiles import get_head_pose,draw_border,iris_center
import numpy as np
import imutils
import time
import dlib
import cv2

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,0)

    for rect in rects:
        (bx,by,bw,bh) = face_utils.rect_to_bb(rect)
        draw_border(frame,(bx,by),(bx+bw,by+bh),(127,255,255),1,10,20)

        shape = predictor(gray,rect)

        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]

        rightEye = shape[rStart:rEnd]



        leftEyeHull = cv2.convexHull(leftEye)

        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, (127, 255, 255), 1)

        cv2.drawContours(frame, [rightEyeHull], -1, (127, 255, 255), 1)

        reprojectdst, euler_angle = get_head_pose(shape)

        image_points = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

        #for start, end in line_pairs:
            #cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 1, (0,0,255), -1)

        #p1 = (int(shape[34][0]), int(shape[34][1]))
        #p2 = (int(reprojectdst[0][0]), int(reprojectdst[0][1]))

        #cv2.line(frame, p1, p2, (255,0,0), 2)

        cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (127, 255, 255), thickness=1)
        cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (127, 255, 255), thickness=1)
        cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (127, 255, 255), thickness=1)

        #cv2.putText(frame,"Left Eye Center is:{}".format(tuple(lefteyecenter)),(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.75, (127, 255, 255), thickness=2)

        #cv2.putText(frame,"Left Eye Center is:{}".format(tuple(righteyecenter)),(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.75, (127, 255, 255), thickness=2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()     