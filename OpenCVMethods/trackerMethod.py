import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Błąd: Nie można otworzyć kamery.")
        return

    success, frame = cap.read()
    if not success:
        print("Błąd: Nie można odczytać klatki z kamery.")
        return

    print("Zaznacz obiekt naciśnij ENTER lub SPACJĘ.")
    print("Aby anulować, naciśnij 'c'.")
    bbox = cv2.selectROI("Wybierz obiekt do sledzenia", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Wybierz obiekt do sledzenia")

    # Sprawdzenie czy użytkownik coś zaznaczył
    if bbox == (0, 0, 0, 0):
        print("Nie wybrano obszaru")
        return

    tracker = None
    tracker = cv2.legacy.TrackerCSRT.create()
    tracker.init(frame, bbox)

    while True:
        success, frame = cap.read()
        if not success:
            break

        tracking_success, box = tracker.update(frame)

        if tracking_success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            """Środek śledzonego obiektu"""
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            cv2.putText(frame, "Status: Sledzenie...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Status: Zgubiono obiekt!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Isolated CSRT Tracker Test", frame)

        # Wyjście klawiszem ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()