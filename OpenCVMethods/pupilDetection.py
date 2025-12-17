import cv2
import numpy as np

global template

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
        dp=1,  # Rozdzielczość akumulatora (1 = taka sama jak obrazu)
        minDist=rows / 8,  # Minimalna odległość między środkami wykrytych okręgów
        param1=edge_detection_threshold,  # Górny próg detektora krawędzi Canny'ego
        param2=accumulator_threshold,  # Próg akumulatora (im mniejszy, tym więcej fałszywych kół wykryje)
        minRadius=int(rows / 10),  # Minimalny promień (żeby nie łapał szumu)
        maxRadius=int(rows / 2)  # Maksymalny promień (żeby nie łapał całego oczodołu)
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


def detect_pupil_template(eye_frame, template):
    """
    Śledzi źrenicę metodą dopasowania wzorca (Template Matching).
    W pierwszej klatce pobiera środek obrazu jako wzorzec.
    W kolejnych klatkach szuka tego wzorca.
    """
    # Kopia obrazu do rysowania wyników
    output_frame = eye_frame.copy()
    h, w, _ = eye_frame.shape

    """Pobranie wzorca jeśli nie ma żadnego"""
    if template is None:
        template_width = int(w * 0.20)
        template_height = int(h * 0.20)

        # Zakładamy, że na początku użytkownik patrzy w środek
        center_x, center_y = w // 2, h // 2

        # Współrzędne wycinka
        x1 = max(0, center_x - template_width // 2)
        y1 = max(0, center_y - template_height // 2)
        x2 = min(w, center_x + template_width // 2)
        y2 = min(h, center_y + template_height // 2)

        # Zapisujemy wzorzec do zmiennej globalnej
        template = eye_frame[y1:y2, x1:x2]

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        return (center_x, center_y), output_frame

    else:
        # Zabezpieczenie przed wzorcem większym niż ROI
        if template.shape[0] > h or template.shape[1] > w:
            template = None
            return None, output_frame

        """Metoda szukania wzorca"""
        res = cv2.matchTemplate(eye_frame, template, cv2.TM_CCOEFF_NORMED)

        # Znajdujemy punkt, gdzie dopasowanie jest najlepsze (max_loc)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # max_loc to lewy górny róg znalezionego obszaru
        top_left = max_loc

        # Obliczamy środek znalezionego obszaru
        h_t, w_t, _ = template.shape
        cx = top_left[0] + w_t // 2
        cy = top_left[1] + h_t // 2

        # Rysujemy zieloną ramkę wokół znalezionego oka
        bottom_right = (top_left[0] + w_t, top_left[1] + h_t)
        cv2.rectangle(output_frame, top_left, bottom_right, (0, 255, 0), 1)

        # Rysujemy kropkę w środku
        cv2.circle(output_frame, (cx, cy), 1, (0, 0, 255), -1)

        return (cx, cy), output_frame


def select_template_manually(frame):
    """
    Otwiera okno, pozwala zaznaczyć wzorzec myszką i zatwierdzić SPACJĄ lub ENTEREM.
    Anulowanie przez 'c'.
    """
    global template
    # selectROI zwraca (x, y, w, h)
    bounding_box = cv2.selectROI("Wybierz oko", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Wybierz oko")

    # Jeśli zaznaczono bounding box
    if bounding_box != (0, 0, 0, 0):
        x, y, w, h = bounding_box
        template = frame[y:y + h, x:x + w]
        print("Wzorzec został pobrany ręcznie!")
    else:
        print("Anulowano wybór wzorca.")

def reset_template():
    """Funkcja pomocnicza do kasowania wzorca z zewnątrz"""
    global template
    template = None
    print("Wzorzec zresetowany - spójrz w środek kamery!")