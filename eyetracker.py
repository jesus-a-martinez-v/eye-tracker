import cv2


class EyeTracker(object):
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def track(self, image):
        face_rectangles = self.face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
                                                             minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        rectangles = []

        for (f_x, f_y, f_w, f_h) in face_rectangles:
            face_roi = image[f_y: f_y + f_h, f_x: f_x + f_w]
            rectangles.append((f_x, f_y, f_x + f_w, f_y + f_h))

            eye_rectangles = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10,
                                                               minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE)

            for (e_x, e_y, e_w, e_h) in eye_rectangles:
                rectangles.append((f_x + e_x, f_y + e_y, f_x + e_x + e_w, f_y + e_y + e_h))

        return rectangles
