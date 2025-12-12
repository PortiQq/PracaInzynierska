import numpy as np
import math

def get_distance(p1, p2):
    """
    Obliczenie odległości między punktami p1 i p2 (euklidesowa)
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_center(landmarks):
    """
    Obliczenie środka punktów
    """
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    return np.mean(x_coords), np.mean(y_coords)
