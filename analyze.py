import pandas as pd
import numpy as np
from pyautogui import size
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os


INPUT_FILE = "data/calibration_data.csv"  # Dane z kalibracji
MODEL_FILE = "data/calibration_model.pkl"  # Plik wyjściowy modelu
SCREEN_WIDTH, SCREEN_HEIGHT = size()

def train():
    if not os.path.exists(INPUT_FILE):
        print(f"BŁĄD: Nie znaleziono pliku {INPUT_FILE}. Przeprowadź kalibrację")
        return

    print(f"--- Wczytywanie danych z {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE)

    # Sprawdzenie liczby unikalnych punktów
    unique_points = df.groupby(['target_x', 'target_y']).size().reset_index().rename(columns={0: 'count'})
    print(f"Znaleziono {len(unique_points)} unikalnych punktów kalibracyjnych:")
    print(unique_points)
    print(f"Łącznie próbek: {len(df)}")

    # Definicja wejścia i wyjścia
    X = df[['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll', 'eye_aspect_ratio']]
    y = df[['target_x', 'target_y']]

    # Podział na dane treningowe i dane walidacyjne
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Trenowanie modelu Random Forest ---")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Zapis modelu do pliku
    joblib.dump(model, MODEL_FILE)
    print(f"Model zapisano do pliku: {MODEL_FILE}")


    """ --- Wizualizacja --- """

    print("Generowanie wykresu podglądowego...")
    predictions = model.predict(X_val)

    plt.figure(figsize=(10, 6))

    # Rysowanie rzeczywistych celów
    plt.scatter(y_val['target_x'], y_val['target_y'], c='red', marker='x', s=100, label='Prawdziwy Cel')

    # Rysowanie przewidywań modelu
    plt.scatter(predictions[:, 0], predictions[:, 1], c='blue', alpha=0.5, label='Predykcja Modelu')

    # Rysowanie linii łączących błąd
    for i in range(len(predictions)):
       plt.plot([y_val.iloc[i]['target_x'], predictions[i,0]], [y_val.iloc[i]['target_y'], predictions[i,1]], 'gray', alpha=0.1)

    plt.title(f"Walidacja treningu: {len(unique_points)} punktów\n")
    plt.xlabel("Pozycja X na ekranie (0.0 - 1.0)")
    plt.ylabel("Pozycja Y na ekranie (0.0 - 1.0)")
    plt.gca().invert_yaxis()  # Odwrócenie osi Y
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == "__main__":
    train()