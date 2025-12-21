import time

from pupilDetection import *
from templateMethod import *

"""Inicjalizacje"""
prev_time = time.time()
template = None
tracker = None
tracker_flag = False

def nothing(x):
    pass

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

"""Switch do wyboru metody śledzenia"""
method = 1

if method == 1:
    cv2.namedWindow("Eye tracking")
    cv2.createTrackbar("Threshold", "Eye tracking", 70, 255, nothing)
elif method == 2:
    cv2.namedWindow("Eye tracking")
    cv2.createTrackbar("Edge detection threshold", "Eye tracking", 40, 255, nothing)
    cv2.createTrackbar("accumulator threshold", "Eye tracking", 25, 255, nothing)
elif method == 3:
    cv2.namedWindow("Eye tracking")
    print(f"Press {"s"} to record eye template")
elif method == 4:
    tracker_flag = True


cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()
    pupil_pos, eye_view = None, None

    """Pomiar FPS"""
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time


    """Wykrycie twarzy z użyciem HaarCascade
       i potraktowanie go jako ROI do ropoznania oczu"""
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]


        """Rozpoznanie obszarów oczu"""
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_color = roi_color[ey:ey + eh, ex:ex + ew]
            eye_roi_copy = eye_color.copy()
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_roi = roi_color[ey:ey + eh, ex:ex + ew]

            if method == 1:
                thresh_val = cv2.getTrackbarPos("Threshold", "Eye tracking")
                pupil_pos, eye_view = detect_pupil(eye_roi, thresh_val)
            elif method == 2:
                param1 = cv2.getTrackbarPos("Edge detection threshold", "Eye tracking")
                param2 = cv2.getTrackbarPos("accumulator threshold", "Eye tracking")
                pupil_pos, eye_view = detect_pupil_hough(eye_roi, param1, param2)
            elif method == 3:
                pupil_pos, eye_view = detect_pupil_template(eye_roi, template)
            elif method == 4:
                if tracker_flag:
                    bbox = cv2.selectROI("Wybierz obiekt do sledzenia", frame, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Wybierz obiekt do sledzenia")
                    if bbox == (0, 0, 0, 0):
                        print("Nie wybrano obszaru. Zamykanie programu.")
                        continue
                    tracker = cv2.legacy.TrackerCSRT.create()
                    tracker.init(frame, bbox)
                    tracker_flag = False
                    eye_view = eye_roi
                else:
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
                    eye_view = eye_roi

            else:
                pupil_pos = False


            bigger = cv2.resize(eye_view, None, fx=5, fy=5, interpolation=cv2.INTER_AREA)

            if pupil_pos:
                px, py = pupil_pos
                cv2.circle(roi_color, (ex + px, ey + py), 1, (0, 0, 255), -1)

                # gaze_ratio = px / ew
                # text = f"{gaze_ratio:.2f}"
                # cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Eye tracking", bigger)


    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    frame = cv2.flip(frame, 1)
    cv2.imshow('Pure OpenCV Eye Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc to quit
        break
    elif key == ord('s') and method == 3:
        eye_roi_copy = cv2.resize(eye_roi_copy, None, fx=5, fy=5, interpolation=cv2.INTER_AREA)
        select_template_manually(eye_roi_copy)

cap.release()
cv2.destroyAllWindows()