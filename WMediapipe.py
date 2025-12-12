from visualize import *

"""Inicjalizacja mediapipowych rzeczy"""
mp_face_mesh = mp.solutions.face_mesh   #468 punktów na twarzy


"""Indeksy obwódki oka bardziej i mniej dokładne
   Pamiętaj że jak obraz jest obrócony to lewe
   oko znajduje się z prawej strony ekranu :* """
LEFT_EYE =[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
# LEFT_EYE=[362, 385, 387, 263, 373, 380]
# RIGHT_EYE =[33, 160, 158, 133, 153, 144]

"""Indeksy lewej i prawej źrenicy"""
LEFT_IRIS = [474, 475, 476, 477]
LEFT_IRIS_CENTER = [473]
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = [468]


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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe używa RGB, a cv2  BGR

        rgb_frame.flags.writeable = False # Dla poprawy wydajności: Sprawdzić?
        result = face_mesh.process(rgb_frame) # Przetwarzanie (inference)
        rgb_frame.flags.writeable = True

        # Logika wyciągania głębokości (tylko jeśli wykryto twarz)
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

            """Rysowanie źrenic i obwódek oczu"""
            final_frame_rgb = draw_landmarks(rgb_frame, result)

            draw_eye_outline(final_frame_rgb, face_landmarks, LEFT_EYE, color=(255, 255, 255), thickness=1)
            draw_eye_outline(final_frame_rgb, face_landmarks, RIGHT_EYE, color=(255, 255, 255), thickness=1)

            draw_eye_center(final_frame_rgb, left_iris_landmarks)
            draw_eye_center(final_frame_rgb, right_iris_landmarks)


            """Spowrotem konwersja na BGR dla OpenCV dla wyświetlenia"""
            final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)
        else:
            """Jeśli nie wykryto twarzy, wyświetlamy czysty obraz
               Żeby się nie wywaliło tak jak wcześniej jak zasłoniłem twarz"""
            final_frame_bgr = frame

        """Wyświetlenie okna z odbiciem w pionie dla lepszej nawigacji (misc)"""
        debug_view = cv2.flip(final_frame_bgr, 1)
        cv2.putText(debug_view, f"L Iris Z: {left_iris_depth:.4f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_view, f"R Iris Z: {right_iris_depth:.4f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Z adnotacjami", debug_view)
        cv2.imshow("Surowy obraz", cv2.flip(frame, 1))


        #cv2.imshow("Bez", cv2.flip(frame,1))
        #cv2.imshow("Z adnotacjami", cv2.flip(final_frame_bgr,1))


        """Esc lub Q - wyłącza program"""
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.waitKey(1) & 0xFF == 27):
            break


"""Koniec"""


webcam.release()
cv2.destroyAllWindows()