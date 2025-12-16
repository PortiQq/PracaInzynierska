from pupilDetection import *

face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow("Eye Threshold")
cv2.createTrackbar("Threshold", "Eye Threshold", 70, 255, nothing)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 10)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)
        for (ex, ey, ew, eh) in eyes:
            eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_color = roi_color[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_roi = roi_color[ey:ey + eh, ex:ex + ew]

            thresh_val = cv2.getTrackbarPos("Threshold", "Eye Threshold")
            pupil_pos, thresh_view = detect_pupil(eye_roi, thresh_val)
            bigger = cv2.resize(thresh_view, None, fx=5, fy=5, interpolation=cv2.INTER_AREA)

            if pupil_pos:
                px, py = pupil_pos
                cv2.circle(roi_color, (ex + px, ey + py), 1, (0, 255, 0), -1)

                gaze_ratio = px / ew
                text = f"{gaze_ratio:.2f}"
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Eye Threshold", bigger)

    cv2.imshow('Pure OpenCV Eye Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()