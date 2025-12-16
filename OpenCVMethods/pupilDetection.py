import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

def detect_pupil(eye_frame, threshold):
    """
    Finds the center of the pupil within an eye frame using thresholding.
    Returns: (cx, cy) of the pupil, or None if not found.
    """
    rows, cols, _ = eye_frame.shape

    gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

    gray_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)

    _, threshold = cv2.threshold(gray_eye, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    # Erozja - usuwięcie szumów i odcięcie cienkich połączeń
    threshold = cv2.erode(threshold, kernel, iterations=1)
    # Dylatacja powiększenie pozostałych elementów (źrenicy)
    threshold = cv2.dilate(threshold, kernel, iterations=1)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    if contours:
        (x, y, w, h) = cv2.boundingRect(contours[0])
        cx = x + int(w / 2)
        cy = y + int(h / 2)
        return (cx, cy), threshold

    return None, threshold

