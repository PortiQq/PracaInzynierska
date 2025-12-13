import numpy as np
from math import sqrt, hypot

def get_distance(p1, p2):
    """
    Obliczenie odległości między punktami p1 i p2 (euklidesowa)
    """
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_center_of_landmarks(landmarks):
    """
    Obliczenie środka punktów
    """
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    return np.mean(x_coords), np.mean(y_coords)


def get_landmark_px(image, landmark):
    h, w, _ = image.shape
    px_values = (int(landmark[0]*w), int(landmark[1]*h))
    return px_values