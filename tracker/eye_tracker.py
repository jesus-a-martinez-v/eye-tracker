import cv2


class EyeTracker(object):
    def __init__(self, face_cascade_path, eye_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def track(self, image):
        face_rectangles = self.face_cascade.detectMultiScale(image,
                                                             scaleFactor=1.1,
                                                             minNeighbors=5,
                                                             minSize=(30, 30),
                                                             flags=cv2.CASCADE_SCALE_IMAGE)

        rectangles = []

        for (f_x, f_y, face_width, face_height) in face_rectangles:
            face_roi = image[f_y: f_y + face_height, f_x: f_x + face_width]
            rectangles.append((f_x, f_y, f_x + face_width, f_y + face_height))

            eye_rectangles = self.eye_cascade.detectMultiScale(face_roi,
                                                               scaleFactor=1.1,
                                                               minNeighbors=10,
                                                               minSize=(20, 20),
                                                               flags=cv2.CASCADE_SCALE_IMAGE)

            for (e_x, e_y, eye_width, eye_height) in eye_rectangles:
                rectangles.append((f_x + e_x, f_y + e_y, f_x + e_x + eye_width, f_y + e_y + eye_height))

        return rectangles
