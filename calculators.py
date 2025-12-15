from useful import *


def get_relative_iris_coords(eye_landmarks, iris_center):
    """
    Oblicza znormalizowaną pozycję źrenicy względem obrysu oka.

    Argumenty:
    eye_landmarks: lista obiektów landmarków (np. punkty konturu oka)
    iris_center: krotka (x, y) oznaczająca środek źrenicy (z get_center_of_landmarks)

    Zwraca:
    (rel_x, rel_y): gdzie (0.5, 0.5) to środek oka, (0,0) lewy górny róg, (1,1) prawy dolny.
    """

    # Wspołrzędne landmarków obrysu oka
    eye_xs = [lm.x for lm in eye_landmarks]
    eye_ys = [lm.y for lm in eye_landmarks]

    # Granice bounding box oka
    min_x = min(eye_xs)
    max_x = max(eye_xs)
    min_y = min(eye_ys)
    max_y = max(eye_ys)

    eye_width = max_x - min_x
    eye_height = max_y - min_y

    if eye_width == 0 or eye_height == 0:   # Dzielenie przez 0
        return 0.5, 0.5

    # Normalizacja: (Wartość - Minimum) / Rozpiętość
    rel_x = (iris_center[0] - min_x) / eye_width
    rel_y = (iris_center[1] - min_y) / eye_height

    return rel_x, rel_y


def get_blink_ratio(landmarks, frame):
    """
    Współczynnik otwarcia oczu (EAR - Eye Aspect Ratio).
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Zwraca średnią wartość dla obu oczu.
    Wartości typowe:
    - Oko otwarte: 0.20 - 0.35
    - Oko zamknięte (mrugnięcie): < 0.20
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

    # Wzór EAR dla lewego oka
    # Dodajemy małą stałą (1e-6) w mianowniku, by uniknąć dzielenia przez zero
    left_ear = (left_eye_vertical_length_1 + left_eye_vertical_length_2) / (2.0 * left_eye_horizontal_length + 1e-6)

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

    right_ear = (right_eye_vertical_length_1 + right_eye_vertical_length_2) / (2.0 * right_eye_horizontal_length + 1e-6)

    ear_avg = (left_ear + right_ear) / 2.0

    return ear_avg


