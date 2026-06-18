import cv2
import numpy as np


def preprocess_image(image):

    # GRAYSCALE
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # BLUR
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # EDGE DETECTION
    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # MORPHOLOGICAL CLOSING
    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # DETECT WALL LINES
    contour_img = image.copy()

    lines = cv2.HoughLinesP(
        closed,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            cv2.line(
                contour_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    return gray, edges, closed, contour_img