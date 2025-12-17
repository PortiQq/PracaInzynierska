import cv2
import numpy as np

global template

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