import cv2
import numpy as np



def detect_pupil(eye_frame, threshold):
    """
    Wykrywa źrenicy/tęczówki przy użyciu progowania
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


def detect_pupil_hough(eye_frame, edge_detection_threshold, accumulator_threshold):
    """
    Wykrywa źrenicę/tęczówkę używając Transformaty Hougha dla okręgów.
    Zwraca: (cx, cy) środka, oraz obraz do wizualizacji (z narysowanym okręgiem).
    """
    output_frame = eye_frame.copy()

    gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

    # Rozmycie - usunięcie szumu z zachowaniem krawędzi
    gray_eye = cv2.medianBlur(gray_eye, 5)

    rows = gray_eye.shape[0]



    # Transformata Hougha
    circles = cv2.HoughCircles(
        gray_eye,
        cv2.HOUGH_GRADIENT,
        dp=1.5,  # Rozdzielczość akumulatora (1 = taka sama jak obrazu)
        minDist=rows / 2,  # Minimalna odległość między środkami wykrytych okręgów
        param1=edge_detection_threshold,  # Górny próg detektora krawędzi Canny'ego
        param2=accumulator_threshold,  # Próg akumulatora (im mniejszy, tym więcej fałszywych kół wykryje)
        minRadius=int(rows / 10),  # Minimalny promień (żeby nie łapał szumu)
        maxRadius=int(rows / 2.5)  # Maksymalny promień (żeby nie łapał całego oczodołu)
    )

    detected_center = None

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # Pierwszy znaleziony okrąg
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            # Zewnętrzny okrąg
            cv2.circle(output_frame, center, radius, (255, 0, 255), 2)
            # Środek koła
            cv2.circle(output_frame, center, 2, (0, 0, 255), 3)

            detected_center = center
            # Przerwanie po wzięciu największego okręgu
            break

    return detected_center, output_frame