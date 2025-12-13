from useful import *

def get_blink_ratio(landmarks, frame):
    """
    Współczynnik otwarcia oczu (EAR - Eye Aspect Ratio).
    Zwraca średnią wartość dla obu oczu.
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
    left_eye_top = ((left_eye_top_1[0] + left_eye_top_2[0]) // 2, (left_eye_top_1[1] + left_eye_top_2[1]) // 2)

    left_eye_bottom_1 = get_coords(385)
    left_eye_bottom_2 = get_coords(387)
    left_eye_bottom = ((left_eye_bottom_1[0] + left_eye_bottom_2[0]) // 2, (left_eye_bottom_1[1] + left_eye_bottom_2[1]) // 2)

    left_eye_horizontal_length = get_distance(left_eye_left, left_eye_right)
    left_eye_vertical_length = get_distance(left_eye_top, left_eye_bottom)

    """Obliczenia dla prawego oka, odległości poziome i pionowe
       środki pionowe jako średnie dwóch punktów"""
    right_eye_left = get_coords(33)
    right_eye_right = get_coords(133)

    right_eye_top_1 = get_coords(160)
    right_eye_top_2 = get_coords(158)
    right_eye_top = ((right_eye_top_1[0] + right_eye_top_2[0]) // 2, (right_eye_top_1[1] + right_eye_top_2[1]) // 2)

    right_eye_bot_1 = get_coords(153)
    right_eye_bot_2 = get_coords(144)
    right_eye_bot = ((right_eye_bot_1[0] + right_eye_bot_2[0]) // 2, (right_eye_bot_1[1] + right_eye_bot_2[1]) // 2)

    right_eye_horizontal_length = get_distance(right_eye_left, right_eye_right)
    right_eye_vertical_length = get_distance(right_eye_top, right_eye_bot)


    left_ratio = left_eye_horizontal_length / (left_eye_vertical_length + 0.001)
    right_ratio = right_eye_horizontal_length / (right_eye_vertical_length + 0.001)
    ratio = (left_ratio + right_ratio) / 2

    return ratio