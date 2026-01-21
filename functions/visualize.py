import cv2
import mediapipe as mp
from functions.utils import *

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def draw_landmarks(image, result):
    """
    Rysowanie źrenic na obrazie z wykorzystaniem
    funkcji dostępnych w MediaPipie bezpośrednio
    """
    image.flags.writeable = True
    if result.multi_face_landmarks:
        for face_landmark in result.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
            )
    return image


def draw_eye_outline(image, landmarks, indices, color=(0, 255, 0), thickness=1):
    """
    Rysuje obwódkę oka na podstawie indeksów punktów.
    Mediapipe ma znormalizowane wartości [0,1] więc
    trzeba zmienić na wartości w pikselach
    """
    points = get_np_array_of_landmarks(image, landmarks, indices)
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)


def draw_eye_center(frame, landmarks, color=(0, 0, 255), thickness=2):
    """
    Zaznacza centrum oka na podstawie indeksów punktów.
    Mediapipe ma znormalizowane wartości [0,1] więc
    trzeba zmienić na wartości w pikselach
    """
    h, w, _ = frame.shape
    # (center_x, center_y) = get_center_of_landmarks(landmarks)
    center_x_px, center_y_px = get_landmark_px(frame, landmarks)
    cv2.circle(frame, (center_x_px, center_y_px), 1, color=color, thickness=thickness)
