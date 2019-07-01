import cv2


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        new_dimension = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        new_dimension = (width, int(h * ratio))

    resized = cv2.resize(image, new_dimension, interpolation=inter)

    return resized
