import cv2

from visualize import *
from calculators import *

"""Inicjalizacja mediapipowych rzeczy"""
mp_face_mesh = mp.solutions.face_mesh   #468 punktów na twarzy

"""Zmienne i flagi"""
font = cv2.FONT_HERSHEY_SIMPLEX

blink_flag = False
BLINK_THRESHOLD  = 10

BINARIZATION_THRESHOLD = 110

"""Indeksy obwódki oka bardziej i mniej dokładne
   lewe oko: obrys od lewej strony dołem do prawej i spowrotem
   prawe oko: obrys od lewej górą do prawej i spowrotem"""
# LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
# RIGHT_EYE= [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LEFT_EYE=[362, 385, 387, 263, 373, 380] # 1pkt po lewej, 2pkt na dole, 1pkt po prawej, 2pkt na górze
RIGHT_EYE =[33, 160, 158, 133, 153, 144] # 1pkt po lewej, 2pkt na górze, 1pkt po prawej, 2pkt na dole

"""Indeksy lewej i prawej źrenicy"""
LEFT_IRIS = [474, 475, 476, 477]
LEFT_IRIS_CENTER = 473
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = 468


"""Główny program"""

webcam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # To dodaje punkty źrenic więc giga ważne
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:

        success, frame = webcam.read() # Odczyt z kamerki
        if not success:
            print("Ignoring empty camera frame")
            continue

        frame_height, frame_width, _ = frame.shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe używa RGB, a cv2  BGR


        rgb_frame.flags.writeable = False
        result = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True


        """Jeśli wykryto twarz na obrazie"""
        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0].landmark

            """Znalezienie tylko źrenic i wartość ich głębokości w obrazie
               I wypisuję sobie na ekranie - może mi się te dane przydadzą"""
            left_eye_landmarks = [face_landmarks[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks[i] for i in RIGHT_EYE]
            left_iris_landmarks = [face_landmarks[i] for i in LEFT_IRIS]    # [474:478]
            right_iris_landmarks = [face_landmarks[i] for i in RIGHT_IRIS]  # [469:473]

            left_iris_depth = left_iris_landmarks[0].z
            right_iris_depth = right_iris_landmarks[0].z

            """Wykrywanie mrugnięć"""
            blink_ratio = get_blink_ratio(face_landmarks, frame)
            if blink_ratio > BLINK_THRESHOLD:
                blink_flag = True
            else:
                blink_flag = False


            """Obliczanie współczynnika spojrzenia z wykorzystaniem
               metody z różnicą odległości landmarków"""
            left_iris_center = get_center_of_landmarks(left_iris_landmarks)
            right_iris_center = get_center_of_landmarks(right_iris_landmarks)
            #left_iris_center = (face_landmarks[LEFT_IRIS_CENTER].x, face_landmarks[LEFT_IRIS_CENTER].y)
            #right_iris_center = (face_landmarks[LEFT_IRIS_CENTER].x, face_landmarks[LEFT_IRIS_CENTER].y)

            gaze_ratio_left_eye = get_gaze_ratio(face_landmarks, LEFT_EYE, left_iris_center)
            gaze_ratio_right_eye = get_gaze_ratio(face_landmarks, RIGHT_EYE, right_iris_center)
            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) /2
            print("gaze_ratio_left_eye", gaze_ratio)



            """Obliczanie współcznnika spojrzenia z wykorzystaniem
               metody z binaryzacją"""
            # gaze_ratio_left_eye = get_gaze_ratio_binarise(frame, face_landmarks, LEFT_EYE, BINARIZATION_THRESHOLD)
            # gaze_ratio_right_eye = get_gaze_ratio_binarise(frame, face_landmarks, RIGHT_EYE, BINARIZATION_THRESHOLD)
            # gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2
            # print(gaze_ratio)



            """##########################################################
               ##########################################################"""
            final_frame_rgb = rgb_frame.copy() # Ostateczny obraz kamery

            """Rysowanie źrenic i obwódek oczu"""
            # draw_landmarks(final_frame_rgb, result)
            #
            draw_eye_outline(final_frame_rgb, face_landmarks, LEFT_EYE, color=(255, 255, 255), thickness=1)
            draw_eye_outline(final_frame_rgb, face_landmarks, RIGHT_EYE, color=(255, 255, 255), thickness=1)
            #
            draw_eye_center(final_frame_rgb, left_iris_landmarks)
            draw_eye_center(final_frame_rgb, right_iris_landmarks)

            """Spowrotem konwersja na BGR dla OpenCV dla ładnego wyświetlenia"""
            final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            final_frame_bgr = frame # Jeśli nie wykryło twarzy -> czysty obraz z kamery


        """Wyświetlenie okna z odbiciem w pionie dla lepszej nawigacji (misc)
           Dodatkowo wypisanie kilku informacji bieżących"""
        debug_view = cv2.flip(final_frame_bgr, 1)

        cv2.putText(debug_view, f"L Iris Z: {left_iris_depth:.4f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_view, f"R Iris Z: {right_iris_depth:.4f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if blink_flag:
            cv2.putText(debug_view, f"Blinking: {blink_flag}", (20, 110), font, 0.6, (0,0,0),2 )

        cv2.imshow("Z adnotacjami", debug_view)
        #cv2.imshow("Surowy obraz", cv2.flip(frame, 1))

        """Esc lub Q - wyłącza program"""
        if cv2.waitKey(1) & 0xFF == 27:
            break


"""Koniec"""


webcam.release()
cv2.destroyAllWindows()