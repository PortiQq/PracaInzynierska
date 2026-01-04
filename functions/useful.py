import numpy as np
from math import hypot

def get_distance(p1, p2):
    """
    Obliczenie odległości między punktami p1 i p2 (euklidesowa)
    :param p1: punkt 1
    :param p2: punkt 2
    return: hypot(p1 - p2)
    """
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def yaw_filter(yaw, yaw_threshold):
    """
    Filtruje szumy kąta yaw przy głowie zwróconej prosto
    :param yaw: kąt obrotu głowy
    :param yaw_threshold: zakres filtrowania
    :return: kąt obrotu po filtracji
    """
    if abs(yaw) < yaw_threshold:
        return pow(yaw, 3)/pow(yaw_threshold, 2)
    else:
        return yaw

def get_center_of_landmarks(landmarks):
    """
    Obliczenie koordynatów środka landmarków
    """
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    center = (np.mean(x_coords), np.mean(y_coords))
    return center


def get_landmark_px(frame, landmark):
    """
    Zwraca koordynaty landmarków w pikselach
    """
    h, w, _ = frame.shape
    px_values = (int(landmark[0]*w), int(landmark[1]*h))
    return px_values


def get_np_array_of_landmarks(frame, landmarks, indices):
    """
    Zamiana listy punktów typu landmark na tablicę np.array
       :returns np.array koordynatów landmarków w pikselach
    """
    points = []
    h, w, _ = frame.shape

    for index in indices:
        lm = landmarks[index]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    return points