from pyautogui import size
from functions.visualize import *
from functions.calculators import *
from functions.fileHandlers import *

FONT = cv2.FONT_HERSHEY_SIMPLEX # Domyślna czcionka

"""Inicjalizacja mediapipowych rzeczy"""
mp_face_mesh = mp.solutions.face_mesh   #468 punktów na twarzy

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

"""Zmienne dot. wykrywania mrugnięć"""
blink_flag = False
BLINK_THRESHOLD  = 0.25

"""Konfiguracja kalibracji
   Punkty kalibracyjne: siatka 3x3 na ekranie 
   (Znormalizowane 0.0-1.0) (x, y)"""
screen_width, screen_height = size()
CALIBRATION_POINTS = [
    (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
    (0.05, 0.5),  (0.5, 0.5),  (0.95, 0.5),
    (0.05, 0.95), (0.5, 0.95), (0.95, 0.95)
]

calibration_flag = False    # Zbieranie danych po naciśnięciu SPACJI
calibration_done = False    # Czy kalibracja zakończona
start_cursor = False        # Wyświetlanie kursora na ramce
calibration_point_index = 0 # Indeks obecnego punktu kalibracyjnego
samples_per_point = 20      # Ile próbek dla jednego punktu
current_samples = 0

# Plik do zapisu danych kalibracyjnych
output_file = "data/default_file.csv"




"""KONFIGURACJA TRYBU PRACY"""
print("Wybierz tryb pracy:")
print("1 - KALIBRACJA (Tworzenie zbioru treningowego)")
print("2 - TESTY (Tworzenie zbioru testowego)")
print("3 - KURSOR (Użycie poprzednich danych kalibracyjnych)")
mode = input("Twój wybór (1/3): ")

calibration_samples = 20
test_samples = 50

if mode == '1':
    output_file = "data/calibration_data.csv" # Zbiór do nauki
    write_header(output_file, 'calibration')
    samples_per_point = calibration_samples
    print(f"Wybrano TRYB KALIBRACJI. Zapis do: {output_file}, Próbek na punkt: {samples_per_point}")
elif mode == '2':
    output_file = "data/test_data.csv"  # Zbiór do weryfikacji
    write_header(output_file, 'calibration')
    samples_per_point = test_samples
    print(f"Wybrano TRYB TESTOWY. Zapis do: {output_file}, Próbek na punkt: {samples_per_point}")
elif mode == "3":
    calibration_done = True                 # Opuszczenie kalibracji
    output_file = "data/session_data.csv"   # Zbiór danych z sesji
    write_header(output_file, 'prediction')
    print(f"Wybrano TRYB DZIAŁANIA. Zapis do: {output_file}")
else:
    output_file = "data/calibration_data.csv"
    write_header(output_file, 'calibration')
    samples_per_point = 20
    print("Nieprawidłowy wybór. Domyślnie tryb KALIBRACJI.")


# Tworzenie okna kalibracji
cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe używa RGB, a cv2  BGR

        """Przetwarzanie obrazu - wykrycie twarzy"""
        rgb_frame.flags.writeable = False
        result = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True


        """Jeśli wykryto twarz na obrazie"""
        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0].landmark

            """Pobranie landmarków obrysu oczu i źrenic"""
            left_eye_landmarks = [face_landmarks[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks[i] for i in RIGHT_EYE]
            left_iris_landmarks = [face_landmarks[i] for i in LEFT_IRIS]    # [474:478]
            right_iris_landmarks = [face_landmarks[i] for i in RIGHT_IRIS]  # [469:473]

            """Wykrywanie mrugnięć"""
            blink_ratio = get_blink_ratio(face_landmarks, frame)
            if blink_ratio < BLINK_THRESHOLD:
                blink_flag = True
            else:
                blink_flag = False

            """Obliczanie współrzędnych spojrzenia 
               relatywnie do kącików oka"""
            left_iris_center = get_center_of_landmarks(left_iris_landmarks)
            right_iris_center = get_center_of_landmarks(right_iris_landmarks)
            l_relative_x, l_relative_y = get_relative_iris_coords(left_eye_landmarks, left_iris_center)
            r_relative_x, r_relative_y = get_relative_iris_coords(right_eye_landmarks, right_iris_center)

            """Obliczenie pozycji głowy"""
            (pitch, yaw, roll), rotation_vector, translation_vector, camera_matrix = get_head_pose(frame, face_landmarks)


            # Wyświetlenie czarnej klatki do rysowania punktów/kursora
            calibration_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            """Przeprowadzanie kalibracji"""
            if not calibration_done:
                if calibration_point_index < len(CALIBRATION_POINTS):
                    point = CALIBRATION_POINTS[calibration_point_index]
                    point_x = int(point[0] * calibration_frame.shape[1])
                    point_y = int(point[1] * calibration_frame.shape[0])

                    # Rysowanie punktu kalibracyjnego (zielony -> zbieranie danych, czerwony -> oczekiwanie na klawisz)
                    color = (0, 255, 0) if calibration_flag else (0, 0, 255)
                    cv2.circle(calibration_frame, (point_x, point_y), 20, color, -1)
                    cv2.putText(calibration_frame, "Patrz na punkt i nacisnij SPACE", (150, 50), FONT, 1, (255, 255, 255), 2)

                    """Zapisywanie danych"""
                    if calibration_flag and not blink_flag:
                        save_calibration_data(output_file, point, l_relative_x, l_relative_y, r_relative_x, r_relative_y,pitch, yaw, roll, blink_ratio)

                        current_samples += 1
                        if current_samples >= samples_per_point:
                            calibration_flag = False
                            current_samples = 0
                            calibration_point_index += 1
                else:
                    calibration_done = True
                    output_file = "data/session_data.csv"
                    write_header(output_file, 'prediction')

                """Jeżeli kalibracja została przeprowadzona
                   rozpoczęcie sesji zbierania danych"""
            else:
                if not start_cursor:
                    cv2.putText(calibration_frame, "Kalibracja zakonczona! ESC -> koniec programu ", (150, 150), FONT, 1,(0, 255, 0), 2)
                    cv2.putText(calibration_frame, "                        SPACE -> pokazanie kursora ", (150, 200), FONT, 1,(0, 255, 0), 2)
                else:
                    save_session_data(output_file, l_relative_x, l_relative_y, r_relative_x, r_relative_y,pitch, yaw, roll, blink_ratio)


            """##########################################################
               # Rysowanie przydatnych informacji na obrazie z kamerki  # 
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

            """Wyświetlenie osi obrotu głowy"""
            cv2.drawFrameAxes(final_frame_rgb, camera_matrix, np.zeros((4, 1)), rotation_vector, translation_vector, length=100,thickness=2)

            """Spowrotem konwersja na BGR dla OpenCV dla ładnego wyświetlenia"""
            final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)

            """Wyświetlenie okna z odbiciem w pionie dla lepszej nawigacji (misc)
                       Dodatkowo wypisanie kilku informacji bieżących"""
            debug_view = cv2.flip(final_frame_bgr, 1)
            cv2.putText(debug_view, f"L Rel: {l_relative_x:.2f}, {l_relative_y:.2f}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(debug_view, f"R Rel: {r_relative_x:.2f}, {r_relative_y:.2f}", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if blink_flag:
                cv2.putText(debug_view, f"Blinking: {blink_flag}", (20, 110), FONT, 0.6, (0, 0, 0), 2)
            if 'pitch' in locals():
                cv2.putText(debug_view, f"Pitch: {pitch:.0f}", (20, 210), FONT, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_view, f"Yaw:   {yaw:.0f}", (20, 250), FONT, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_view, f"Roll:  {roll:.0f}", (20, 290), FONT, 0.6, (0, 255, 255), 2)

        else:
            final_frame_bgr = frame # Jeśli nie wykryło twarzy -> czysty obraz z kamery


        """Wyświetlenie okien"""
        cv2.imshow("Z adnotacjami", debug_view)
        cv2.imshow("Calibration", calibration_frame)


        """Obsługa klawiszy:
           Spacja - zaczyna pobierać próbki kalibracji/rozpoczyna ruch kursora
           t - zaczyna zbieranie nowych danych testowych do walidacji
           Esc - wyłącza program """
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            calibration_flag = True
            if calibration_done:
                start_cursor = True
        elif key == ord('t'):
            if calibration_done:
                calibration_flag = False
                calibration_done = False
                start_cursor = False
                calibration_point_index = 0
                samples_per_point = test_samples
                current_samples = 0
                output_file = "data/test_data.csv"
                print(f"Wybrano TRYB TESTOWY. Zapis do: {output_file}, Próbek na punkt: {samples_per_point}")
                write_header(output_file, 'calibration')
        elif key == 27:
            break


"""Koniec działania programu"""
webcam.release()
cv2.destroyAllWindows()