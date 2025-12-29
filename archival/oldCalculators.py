import cv2
import numpy as np
from functions.useful import get_distance, get_np_array_of_landmarks


def find_farthest_landmark_index(landmarks, indices_list, direction):
    """
    Znalezienie indeksu punktu wysuniętego najbardziej w lewo lub w prawo
    direction: 'left' / 'right'
    :returns index of farthest landmark
    !!!Przydatne tylko przy zmienianiu listy landmarków!!!
    """
    if direction == 'left':
        return min(indices_list, key=lambda i: landmarks[i].x)

    elif direction == 'right':
        return max(indices_list, key=lambda i: landmarks[i].x)

    else:
        raise ValueError("Direction must be 'left' or 'right'")


def get_gaze_ratio(face_landmarks, eye_points, iris_center):
    """
    eye_points: [] tablica punktów obrysu oka
    iris_center: (x, y) środka źrenicy
    """
    left_corner_index = find_farthest_landmark_index(face_landmarks, eye_points, "left")
    right_corner_index = find_farthest_landmark_index(face_landmarks, eye_points, "right")

    left_corner = (face_landmarks[left_corner_index].x, face_landmarks[left_corner_index].y)
    right_corner = (face_landmarks[right_corner_index].x, face_landmarks[right_corner_index].y)

    dist_to_left_corner = get_distance(left_corner, iris_center)
    dist_to_right_corner = get_distance(right_corner, iris_center)

    # Zabezpieczenie przed dzieleniem przez zero
    if dist_to_right_corner == 0:
        return 1.0

    ratio = dist_to_left_corner / dist_to_right_corner
    return ratio


def get_gaze_ratio_binarise(frame, face_landmarks, eye_points, binarisation_threshold):
    """Wydzielenie obszarów oczu i wyodrębnienie źrenic
       z wykorzystaniem binaryzacji obrazu"""

    frame_height, frame_width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_region = get_np_array_of_landmarks(frame, face_landmarks, eye_points)

    # Nałożenie maski, żeby wyodrębnić dokładny kontur oka
    mask = np.zeros((frame_height, frame_width), np.uint8)
    cv2.polylines(mask, [eye_region], True, (255, 255, 255), 2)
    cv2.fillPoly(mask, [eye_region], (255, 255, 255))
    eye_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    # Wydzielenie konturów regionu oka
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # Wydzielenie i binaryzacja obrazu oka
    gray_eye_frame = eye_frame[min_y:max_y, min_x:max_x]
    gray_eye_frame = cv2.GaussianBlur(gray_eye_frame, (3, 3), 0)  # Filtr gaussa przy słabej jakości obrazu
    _, threshold_eye = cv2.threshold(gray_eye_frame, binarisation_threshold, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    gaze_ratio = left_side_white / (right_side_white + 0.001)

    """Wizualizacja (tylko dla podglądu co się dzieje"""
    # threshold_eye = cv2.resize(threshold_eye, None, fx=7, fy=7)
    # eye_frame = cv2.resize(gray_eye_frame, None, fx=5, fy=5)
    # left_side_threshold = cv2.resize(left_side_threshold, None, fx=5, fy=5)
    # right_side_threshold = cv2.resize(right_side_threshold, None, fx=5, fy=5)
    # cv2.imshow("Podgląd oka (gray)", eye_frame)
    # cv2.imshow("Oko po binaryzacji", threshold_eye)
    # cv2.imshow("Lewe oko lewa strona", left_side_threshold)
    # cv2.imshow("Lewe oko Prawa strona", right_side_threshold)

    return gaze_ratio