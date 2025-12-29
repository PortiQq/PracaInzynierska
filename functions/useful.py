import numpy as np
from math import hypot

def get_distance(p1, p2):
    """
    Obliczenie odległości między punktami p1 i p2 (euklidesowa)
    """
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_center_of_landmarks(landmarks):
    """
    Obliczenie koordynatów środka punktów
    """
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    center = (np.mean(x_coords), np.mean(y_coords))
    return center


def get_landmark_px(frame, landmark):
    h, w, _ = frame.shape
    px_values = (int(landmark[0]*w), int(landmark[1]*h))
    return px_values


def get_np_array_of_landmarks(frame, landmarks, indices):
    """Zamiana listy punktów typu landmark na tablicę np.array
       :returns np.array of points coordinates in pixels"""
    points = []
    h, w, _ = frame.shape

    for index in indices:
        lm = landmarks[index]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    return points