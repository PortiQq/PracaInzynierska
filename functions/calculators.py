from functions.useful import *
import cv2


def get_relative_iris_coords(eye_landmarks, iris_center, corner_landmark):
    """
    Oblicza znormalizowaną pozycję źrenicy względem kącików oka.

    Argumenty:
    eye_landmarks: lista landmarków (np. punkty konturu oka)
    iris_center: krotka (x, y) oznaczająca środek źrenicy (z get_center_of_landmarks)

    Zwraca:
    (rel_x, rel_y)
    """

    # Wspołrzędne landmarków obrysu oka
    eye_xs = [lm.x for lm in eye_landmarks]
    eye_ys = [lm.y for lm in eye_landmarks]

    # Kąciki oka
    min_x = min(eye_xs)
    max_x = max(eye_xs)

    corner_y = corner_landmark.y

    eye_width = max_x - min_x
    if eye_width == 0:
        return 0.5, 0.5

    # środek Y -> średnia Y z punktów o min_x i max_x
    corner_1_y = eye_ys[eye_xs.index(min_x)]
    corner_2_y = eye_ys[eye_xs.index(max_x)]
    center_y = (corner_1_y + corner_2_y) / 2.0

    """relatywne położenie centrum źrenicy - współrzędna X
       od lewego kącika do środka źrenicy
       normalizacja względem szerokości oka"""
    rel_x = (iris_center[0] - min_x) / eye_width

    """relatywne położenie centrum źrenicy - współrzędna Y
       normalizacja względem szerokości oka (wysokość zmienia się przy patrzenie góra - dół)"""
    rel_y = (iris_center[1] - corner_y) / eye_width

    return rel_x, rel_y



def get_head_pose(frame, face_landmarks):
    """
    Oblicza orientację głowy (Pitch, Yaw, Roll) wykorzystując algorytm PnP.
    Zwraca kąty w stopniach oraz wektor rotacji i translacji (do wizualizacji).
    """
    height, width, _ = frame.shape

    # Standardowy model 3D twarzy
    # Współrzędne w milimetrach, arbitralne
    face_3d = np.array([
        [0.0, 0.0, 0.0],  # Nose tip (4)
        [0.0, -330.0, -65.0],  # Chin (152)
        [-225.0, 170.0, -135.0],  # Left eye left corner (33)
        [225.0, 170.0, -135.0],  # Right eye right corner (263)
        [-150.0, -150.0, -125.0],  # Left Mouth corner (61)
        [150.0, -150.0, -125.0]  # Right mouth corner (291)
    ], dtype=np.float64)


    nose_left_lm = face_landmarks[48]
    nose_right_lm = face_landmarks[278]

    nose_l_x, nose_l_y = nose_left_lm.x * width, nose_left_lm.y * height
    nose_r_x, nose_r_y = nose_right_lm.x * width, nose_right_lm.y * height

    avg_nose_x = (nose_l_x + nose_r_x) / 2
    avg_nose_y = (nose_l_y + nose_r_y) / 2

    face_2d = []
    # Nos jako średnia dwóch punktów
    face_2d.append([avg_nose_x, avg_nose_y])
    # Pozostałe punkty
    other_landmarks = [152, 33, 263, 61, 291]
    for idx in other_landmarks:
        lm = face_landmarks[idx]
        x, y = int(lm.x * width), int(lm.y * height)
        face_2d.append([x, y])
    face_2d = np.array(face_2d, dtype=np.float64)

    # Macierz kamery (przybliżona)
    # TODO: wczytanie danych kalibracji kamery przed rozpoczęciem programu???
    focal_length = 1 * width
    camera_matrix = np.array([
        [focal_length, 0, height / 2],
        [0, focal_length, width / 2],
        [0, 0, 1]
    ])

    # Macierz dystorsji (jako 0 dla uproszczenia)
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, camera_matrix, distortion_matrix)

    # Konwersja wektora rotacji na kąty Eulera
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    # angles: [pitch, yaw, roll]
    x = angles[0]
    y = angles[1]
    z = angles[2]

    """Korekta kąta pitch - przesunięcie wartości o 180 stopni
       oraz odwrócenie osi pitch: ujemne -> dół, dodatnie -> góra
       oraz roll: ujemne -> w lewo, dodatnie -> w prawo """
    if x > 0:
        x -= 180
    else:
        x += 180
    x = -x
    z = -z

    return (x, y, z), rotation_vector, translation_vector, camera_matrix



def get_blink_ratio(landmarks, frame):
    """
    Współczynnik otwarcia oczu (EAR - Eye Aspect Ratio).
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Zwraca średnią wartość dla obu oczu.
    Wartości typowe:
    - Oko otwarte: 0.20 - 0.35
    - Oko zamknięte: < 0.20
    """
    h, w, _ = frame.shape

    def get_coords(index):
        """Oblicza współrzędne pikselowe dla danego punktu
           czyli podanego landmarków (po indeksie)"""
        return int(landmarks[index].x * w), int(landmarks[index].y * h)


    """Obliczenia dla lewego oka, odległości poziome i pionowe
       środki pionowe jako średnie dwóch punktów"""
    left_eye_left = get_coords(362)
    left_eye_right = get_coords(263)

    left_eye_top_1 = get_coords(380)
    left_eye_top_2 = get_coords(373)

    left_eye_bottom_1 = get_coords(385)
    left_eye_bottom_2 = get_coords(387)

    left_eye_vertical_length_1 = get_distance(left_eye_top_1, left_eye_bottom_1)
    left_eye_vertical_length_2 = get_distance(left_eye_top_2, left_eye_bottom_2)
    left_eye_horizontal_length = get_distance(left_eye_left, left_eye_right)

    # Eye Aspect Ratio lewego oka
    left_ear = (left_eye_vertical_length_1 + left_eye_vertical_length_2) / (2.0 * left_eye_horizontal_length + 1e-6)  # + 1e-6 dla uniknięcia dzielenia przez zero

    """Obliczenia dla prawego oka, odległości poziome i pionowe
       środki pionowe jako średnie dwóch punktów"""
    right_eye_left = get_coords(33)
    right_eye_right = get_coords(133)

    right_eye_top_1 = get_coords(160)
    right_eye_top_2 = get_coords(158)

    right_eye_bot_1 = get_coords(153)
    right_eye_bot_2 = get_coords(144)

    right_eye_vertical_length_1 = get_distance(right_eye_top_1, right_eye_bot_1)
    right_eye_vertical_length_2 = get_distance(right_eye_top_2, right_eye_bot_2)
    right_eye_horizontal_length = get_distance(right_eye_left, right_eye_right)

    # Eye Aspect Ratio lewego oka
    right_ear = (right_eye_vertical_length_1 + right_eye_vertical_length_2) / (2.0 * right_eye_horizontal_length + 1e-6)

    ear_avg = (left_ear + right_ear) / 2.0

    return ear_avg


