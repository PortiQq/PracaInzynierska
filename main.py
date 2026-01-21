import time
import joblib
import pandas as pd
from pyautogui import size
from collections import deque
from functions.visualize import *
from functions.calculators import *
from functions.fileHandlers import *
from functions.train import train


"""Globalne zmienne dot. ekranu, debugu, plików"""
SCREEN_WIDTH, SCREEN_HEIGHT = size()
FONT = cv2.FONT_HERSHEY_SIMPLEX
CALIBRATION_FILE = "data/calibration_data.csv"
VALIDATION_FILE = "data/validation_data.csv"
SESSION_FILE = "data/session_data.csv"
MODEL_FILE = "data/calibration_model.pkl"
current_output_file = "data/default_file.csv"


"""Inicjalizacja mediapipe mesh - 468 punktów na twarzy"""
mp_face_mesh = mp.solutions.face_mesh


"""Indeksy obrysu oka
   lewe oko: od lewego kącika poprzez dolne punkty
   prawe oko: od lewego kącika poprzez górne punkty"""
LEFT_EYE=[362, 385, 387, 263, 373, 380]
RIGHT_EYE =[33, 160, 158, 133, 153, 144]
LEFT_EYE_RIGHT_CORNER = 263
RIGHT_EYE_LEFT_CORNER = 33

"""Indeksy lewej i prawej tęczówki oraz środka źrenicy"""
LEFT_IRIS = [474, 475, 476, 477]
LEFT_IRIS_CENTER = 473
RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = 468

"""Zmienne dot. zakresów (+mruganie)"""
blink_flag = False
BLINK_THRESHOLD  = 0.275
YAW_THRESHOLD = 15

"""Konfiguracja kalibracji
   Punkty kalibracyjne: siatka 3x3 na ekranie 
   (Znormalizowane 0.0-1.0) (x, y)"""
CALIBRATION_POINTS = [
    (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
    (0.05, 0.5),  (0.5, 0.5),  (0.95, 0.5),
    (0.05, 0.95), (0.5, 0.95), (0.95, 0.95)
]
take_calibration_sample = False    # Zbieranie danych do kalibracji po naciśnięciu SPACJI
calibration_done = False           # Czy kalibracja zakończona

validation_flag = False            # Zbieranie danych testowych
validation_done = False            # Czy testowanie zakończone
take_validation_sample = False     # Zbieranie danych do walidacji po naciśnięciu SPACJI

current_point_index = 0     # Indeks aktualnego punktu kalibracyjnego/testowego
samples_per_point = 0       # Ile próbek dla jednego punktu
current_samples = 0         # Zebrane próbki dla aktualnego punktu
CALIBRATION_SAMPLES = 100   # Liczba punktów dla danych kalibracyjnych
VALIDATION_SAMPLES = 30     # Liczba punktów dla danych testowych

VALIDATION_POINTS = [
    (0.05, 0.05), (0.5, 0.05), (0.95, 0.05),
    (0.05, 0.5),  (0.5, 0.5),  (0.95, 0.5),
    (0.05, 0.95), (0.5, 0.95), (0.95, 0.95)
]

YAW_ALPHA = 0.1
filtered_yaw = 0

"""Inicjalizacja modelu uczenia"""
model = None
model_trained = False


"""Konfiguracja wygładzania ruchu kursora (Smoothing)"""
start_cursor = False                            # Wyświetlanie kursora na ramce
SMOOTHING_BUFFER_SIZE = 15                      # Średnia z ostatnich X klatek
x_buffer = deque(maxlen=SMOOTHING_BUFFER_SIZE)  # Kolejka dwustronna
y_buffer = deque(maxlen=SMOOTHING_BUFFER_SIZE)  # jako bufor dla predykcji


""" --- KONFIGURACJA TRYBU PRACY --- """

print("Wybierz tryb pracy:")
print("1 - KALIBRACJA (Tworzenie zbioru treningowego)")
print("2 - MENU (Użycie poprzednich danych kalibracyjnych)")
mode = input("Twój wybór (1/2): ")
if mode == '1':
    current_output_file = CALIBRATION_FILE # Zbiór do nauki
    write_header(current_output_file, 'calibration')
    samples_per_point = CALIBRATION_SAMPLES
    print(f"Wybrano TRYB KALIBRACJI. Zapis do: {current_output_file}, Próbek na punkt: {samples_per_point}")
elif mode == "2":
    calibration_done = True
    try:
        model = joblib.load(MODEL_FILE)
        print("Model załadowany pomyślnie.")
        model_trained = True
    except FileNotFoundError:
        print("Błąd: Nie znaleziono modelu po treningu.")
    current_output_file = SESSION_FILE   # Zbiór danych z sesji
    write_header(current_output_file, 'session')
    print(f"Wybrano TRYB BEZ KALIBRACJI. Zapis do: {current_output_file}")
else:
    current_output_file = CALIBRATION_FILE
    write_header(current_output_file, 'calibration')
    samples_per_point = CALIBRATION_SAMPLES
    print("Nieprawidłowy wybór. Domyślnie tryb KALIBRACJI.")

# Tworzenie okna kalibracji
cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

fps_buffer = deque(maxlen=60) # Średnia z ostatnich 30 klatek
prev_time = time.time()

"""Pętla główna programu"""

webcam = cv2.VideoCapture(0)

# refine_landmarks - dołączenie modelu MediaPipe Iris
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
) as face_mesh:

    while True:

        success, frame = webcam.read() # Odczyt klatki obrazu z kamery
        if not success:
            print("Zignorowano pustą klatkę obrazu")
            continue


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe używa RGB, a cv2  BGR

        """Przetwarzanie obrazu - wykrycie twarzy"""
        rgb_frame.flags.writeable = False
        result = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Wyświetlenie czarnego okna do kalibracji/testowania
        calibration_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)


        """Jeśli wykryto twarz na obrazie"""
        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0].landmark

            """Pobranie landmarków obrysu oczu i źrenic"""
            left_eye_landmarks = [face_landmarks[i] for i in LEFT_EYE]
            right_eye_landmarks = [face_landmarks[i] for i in RIGHT_EYE]
            left_iris_landmarks = [face_landmarks[i] for i in LEFT_IRIS]    # [474:478]
            right_iris_landmarks = [face_landmarks[i] for i in RIGHT_IRIS]  # [469:473]

            # left_iris_center = get_center_of_landmarks(left_iris_landmarks)
            left_iris_center = face_landmarks[LEFT_IRIS_CENTER]
            # right_iris_center = get_center_of_landmarks(right_iris_landmarks)
            right_iris_center = face_landmarks[RIGHT_IRIS_CENTER]

            """Wykrywanie mrugnięć"""
            blink_ratio = get_blink_ratio(face_landmarks, frame)
            if blink_ratio < BLINK_THRESHOLD:
                blink_flag = True
            else:
                blink_flag = False

            """Obliczanie współrzędnych źrenicy 
               relatywnie do kącików oka"""
            l_relative_x, l_relative_y = get_relative_iris_coords(
                left_eye_landmarks, left_iris_center, face_landmarks[LEFT_EYE_RIGHT_CORNER])
            r_relative_x, r_relative_y = get_relative_iris_coords(
                right_eye_landmarks, right_iris_center, face_landmarks[RIGHT_EYE_LEFT_CORNER])

            """Obliczenie pozycji głowy"""
            (pitch, yaw, roll), rotation_vector, translation_vector, camera_matrix = get_head_pose(frame, face_landmarks)
            # yaw = yaw_filter(yaw, YAW_THRESHOLD)    # Filtrowanie szumu kąta yaw

            # Zastosowanie filtra EMA
            # Nowa wartość to średnia ważona 10% obecnego obrotu i 90% historii (gdy alpha=0.1)
            filtered_yaw = filtered_yaw * (1 - YAW_ALPHA) + yaw * YAW_ALPHA

            # Użycie wartości po przejściu przez filtr
            yaw = filtered_yaw

            """Pomiar FPS"""
            new_time = time.time()
            time_diff = new_time - prev_time
            prev_time = new_time

            if time_diff > 0:
                curr_fps = 1.0 / time_diff
                fps_buffer.append(curr_fps)
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                fps_text = f"FPS: {int(avg_fps)}"

            """Przeprowadzanie kalibracji jeżeli nie została przeprowadzona wcześniej"""
            if not calibration_done:
                if current_point_index < len(CALIBRATION_POINTS):
                    point = CALIBRATION_POINTS[current_point_index]
                    point_x = int(point[0] * calibration_frame.shape[1])
                    point_y = int(point[1] * calibration_frame.shape[0])

                    # Rysowanie punktu kalibracyjnego (zielony -> zbieranie danych, czerwony -> oczekiwanie na klawisz)
                    color = (0, 255, 0) if take_calibration_sample else (0, 0, 255)
                    cv2.circle(calibration_frame, (point_x, point_y), 20, color, -1)
                    cv2.putText(calibration_frame, "Utrzymuj wzrok na punkcie i nacisnij SPACE", (150, 50), FONT, 1, (255, 255, 255), 2)
                    cv2.putText(calibration_frame, "Dodaj ruchy glowy w naturalnym zakresie", (150, 100), FONT, 1, (255, 255, 255), 2)

                    """Zapisywanie danych dot. bieżącego punktu kalibracyjnego"""
                    if take_calibration_sample and not blink_flag:
                        save_calibration_data(current_output_file, point, l_relative_x, l_relative_y, r_relative_x, r_relative_y,pitch, yaw, roll, blink_ratio)

                        current_samples += 1
                        if current_samples >= samples_per_point:
                            take_calibration_sample = False
                            current_samples = 0
                            current_point_index += 1    # Przejście do kolejnego punktu
                else:
                    calibration_done = True
                    current_output_file = "data/session_data.csv"
                    write_header(current_output_file, 'session')

                """Jeżeli kalibracja została przeprowadzona
                   trenowanie modelu regresji - jeden raz
                   rozpoczęcie sesji zbierania danych
                   SPACE - początek zbierania danych"""
            else:
                if not model_trained:
                    print("Rozpoczynam trenowanie modelu...")
                    train(visualise=False)
                    try:
                        model = joblib.load(MODEL_FILE)
                        print("Model załadowany pomyślnie.")
                        model_trained = True
                    except FileNotFoundError:
                        print("Błąd: Nie znaleziono modelu po treningu.")

                """Po wytrenowaniu modelu wyświetlenie menu kontekstowego
                   SPACE -> rozpoczęcie predykcji na żywo i ruch kursora"""
                if start_cursor:
                    input_data = pd.DataFrame([[
                        # l_relative_x, l_relative_y, r_relative_x, r_relative_y,pitch, yaw, roll,]],
                        l_relative_x, l_relative_y, r_relative_x, r_relative_y, pitch, yaw]],
                        columns=['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw'])
                        # columns=['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll'])
                    # Predykcja i dodanie predykcji do bufora
                    prediction = model.predict(input_data)
                    pred_x, pred_y = prediction[0]
                    x_buffer.append(pred_x)
                    y_buffer.append(pred_y)

                    """Wygładzanie za pomocą średniej ruchomej"""
                    mean_x = np.mean(list(x_buffer))
                    mean_y = np.mean(list(y_buffer))
                    # Konwersja na piksele
                    screen_x = int(mean_x * SCREEN_WIDTH)
                    screen_y = int(mean_y * SCREEN_HEIGHT)
                    # Zabezpieczenie krawędzi
                    screen_x = max(0, min(screen_x, SCREEN_WIDTH))
                    screen_y = max(0, min(screen_y, SCREEN_HEIGHT))

                    """Wizualizacja kursora"""
                    # Kursor
                    cv2.circle(calibration_frame, (screen_x, screen_y), 15, (0, 0, 255), -1)
                    # Współrzędne
                    cv2.putText(calibration_frame, f"X: {screen_x} Y: {screen_y}", (screen_x + 20, screen_y), FONT, 0.5,(0, 0, 255), 1)
                    # Kropka pokazująca wszystkie punkty predykcji - bez uśredniania
                    raw_screen_x = int(pred_x * SCREEN_WIDTH)
                    raw_screen_y = int(pred_y * SCREEN_HEIGHT)
                    cv2.circle(calibration_frame, (raw_screen_x, raw_screen_y), 5, (0, 255, 255), -1) # Żółta mała kropka

                    # zapisywanie predykcji w czasie
                    current_time = time.time()
                    save_session_data(current_output_file,current_time,screen_x,screen_y)


                    """ >>> T <<<
                    Po wytrenowaniu modelu zbieranie danych
                       do walidacji wciskając T """
                elif validation_flag:
                    if current_point_index < len(VALIDATION_POINTS):
                        point = VALIDATION_POINTS[current_point_index]
                        point_x = int(point[0] * calibration_frame.shape[1])
                        point_y = int(point[1] * calibration_frame.shape[0])

                        # Rysowanie punktu kalibracyjnego (zielony -> zbieranie danych, czerwony -> oczekiwanie na klawisz)
                        color = (0, 255, 0) if take_validation_sample else (0, 0, 255)
                        cv2.circle(calibration_frame, (point_x, point_y), 20, color, -1)
                        cv2.putText(calibration_frame, "Patrz na punkt i nacisnij SPACE", (150, 50), FONT, 1,
                                    (255, 255, 255), 2)


                        """Po naciśnięciu SPACJI Zapisywanie danych dot. bieżącego punktu"""
                        if take_validation_sample and not blink_flag:
                            input_data = pd.DataFrame([[
                                # l_relative_x, l_relative_y, r_relative_x, r_relative_y, pitch, yaw, roll,
                                l_relative_x, l_relative_y, r_relative_x, r_relative_y, pitch, yaw,
                            ]],
                                # columns=['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll'])
                                columns=['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw'])
                            # Predykcja i dodanie predykcji do bufora
                            prediction = model.predict(input_data)
                            pred_x, pred_y = prediction[0]
                            x_buffer.append(pred_x)
                            y_buffer.append(pred_y)

                            """Wygładzanie za pomocą średniej ruchomej"""
                            mean_x = np.mean(list(x_buffer))
                            mean_y = np.mean(list(y_buffer))
                            # Konwersja na piksele
                            screen_x = int(mean_x * SCREEN_WIDTH)
                            screen_y = int(mean_y * SCREEN_HEIGHT)
                            # Zabezpieczenie krawędzi
                            screen_x = max(0, min(screen_x, SCREEN_WIDTH))
                            screen_y = max(0, min(screen_y, SCREEN_HEIGHT))

                            # Kursor
                            cv2.circle(calibration_frame, (screen_x, screen_y), 15, (255, 255, 255), -1)

                            # Zapisywanie uśrednionych predykcji do pliku (gdy zapełniono bufor)
                            if current_samples >= SMOOTHING_BUFFER_SIZE:
                                save_validation_data(current_output_file, point_x, point_y, screen_x,screen_y)

                            current_samples += 1
                            if current_samples >= samples_per_point + SMOOTHING_BUFFER_SIZE:
                                take_validation_sample = False
                                current_samples = 0
                                current_point_index += 1  # Przejście do kolejnego punktu
                    else:
                        validation_flag = False

                    """Menu kontekstowe -> czekanie na decyzję"""
                else:
                    cv2.putText(calibration_frame, "Kalibracja zakonczona! SPACE -> Zbieranie danych i wizualizacja kursora" , (150, 150), FONT, 1,(0, 255, 0), 2)
                    cv2.putText(calibration_frame, "                        T -> Zbieranie danych do ewaluacji systemu  ", (150, 200), FONT, 1,(0, 255, 0), 2)
                    cv2.putText(calibration_frame, "                        ESC -> koniec programu ", (150, 250), FONT, 1,(0, 255, 0), 2)


            """##########################################################
               # Rysowanie przydatnych informacji na obrazie z kamerki  # 
               ##########################################################"""
            final_frame_rgb = rgb_frame.copy() # Ostateczny obraz kamery

            """Rysowanie źrenic i obwódek oczu"""
            # draw_landmarks(final_frame_rgb, result)
            #
            draw_eye_outline(final_frame_rgb, face_landmarks, LEFT_EYE, color=(255, 0, 0), thickness=1)
            draw_eye_outline(final_frame_rgb, face_landmarks, RIGHT_EYE, color=(255, 0, 0), thickness=1)
            #
            draw_eye_center(final_frame_rgb, left_iris_center)
            draw_eye_center(final_frame_rgb, right_iris_center)

            """Wyświetlenie osi obrotu głowy"""
            cv2.drawFrameAxes(final_frame_rgb, camera_matrix, np.zeros((4, 1)), rotation_vector, translation_vector, length=100,thickness=2)



            """Spowrotem konwersja na BGR dla OpenCV dla ładnego wyświetlenia"""
            final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)


            """Wyświetlenie okna z odbiciem w pionie dla lepszej nawigacji (misc)
                       Dodatkowo wypisanie kilku informacji bieżących"""
            debug_view = cv2.flip(final_frame_bgr, 1)
            debug_view = cv2.resize(debug_view, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
            cv2.putText(debug_view, f"Polozenie lewej zrenicy: {l_relative_x:.2f}, {l_relative_y:.6f}", (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(debug_view, f"Polozenie prawej zrenicy: {r_relative_x:.2f}, {r_relative_y:.6f}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if blink_flag:
                cv2.putText(debug_view, f"Wykryto mrugniecie: {blink_flag}", (20, 100), FONT, 1, (0, 0, 0), 2)
            if 'pitch' in locals():
                cv2.putText(debug_view, f"Pochylenie: {pitch:.0f}", (20, 220), FONT, 0.8, (0, 255, 255), 2)
                cv2.putText(debug_view, f"Odchylenie:   {yaw:.0f}", (20, 260), FONT, 0.8, (0, 255, 255), 2)
                cv2.putText(debug_view, f"Przechylenie:  {roll:.0f}", (20, 300), FONT, 0.8, (0, 255, 255), 2)

            """Wyświetlenie liczby FPS"""
            # cv2.putText(debug_view, fps_text, (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            debug_view = frame # Jeśli nie wykryło twarzy -> czysty obraz z kamery


        """Wyświetlenie okien"""
        cv2.imshow("Podglad obrazu z kamery", debug_view)
        cv2.imshow("Calibration", calibration_frame)


        """Obsługa klawiszy:
           Spacja - zaczyna pobierać próbki kalibracji/rozpoczyna ruch kursora
           t - zaczyna zbieranie nowych danych testowych do walidacji
           Esc - wyłącza program """
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if not calibration_done:
                take_calibration_sample = True
            elif validation_flag:
                take_validation_sample = True
            elif not start_cursor:
                current_output_file = SESSION_FILE
                write_header(current_output_file, 'session')
                start_cursor = True
            else:
                continue
        elif key == ord('t'):
            if calibration_done:
                validation_flag = True
                start_cursor = False
                current_point_index = 0
                samples_per_point = VALIDATION_SAMPLES
                current_samples = 0
                current_output_file = VALIDATION_FILE
                print(f"Wybrano TRYB TESTOWY. Zapis do: {current_output_file}, Próbek na punkt: {samples_per_point}")
                write_header(current_output_file, 'validation')
        elif key == 27:
            break


"""Koniec działania programu"""
webcam.release()
cv2.destroyAllWindows()