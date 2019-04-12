import argparse
import cv2
import imutils
from eyetracker import EyeTracker

argparser = argparse.ArgumentParser()
argparser.add_argument('-f', '--face', required=True, help='Path to where the face cascade resides.')
argparser.add_argument('-e', '--eye', required=True, help='Path to where the eye cascade resides.')
argparser.add_argument('-v', '--video', help='Path to where the (optional) video file resides.')
arguments = vars(argparser.parse_args())

eye_tracker = EyeTracker(arguments['face'], arguments['eye'])

if not arguments.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(arguments['video'])

while True:
    (grabbed, frame) = camera.read()

    if arguments.get('video') and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rectangles = eye_tracker.track(gray)

    green = (0, 255, 0)
    thickness = 2
    for rectangle in rectangles:
        x1, y1, x2, y2 = rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), green, thickness)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()