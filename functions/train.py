import pandas as pd
import numpy as np
from pyautogui import size
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

MODEL_FILE = "../data/calibration_model.pkl"
INPUT_FILE = "../data/calibration_data.csv"
SCREEN_WIDTH, SCREEN_HEIGHT = size()

def train(visualise = True):
    if not os.path.exists(INPUT_FILE):
        print(f"BŁĄD: Nie znaleziono pliku {INPUT_FILE}. Przeprowadź kalibrację")
        return
    print(f"--- Wczytywanie danych z {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE)

    # Sprawdzenie liczby unikalnych punktów kalibracyjnych
    unique_points = df.groupby(['target_x', 'target_y']).size().reset_index().rename(columns={0: 'count'})
    print(f"Znaleziono {len(unique_points)} unikalnych punktów:")
    print(unique_points)
    print(f"Łącznie próbek: {len(df)}")

    # Definicja wejścia i wyjścia
    # X = df[['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll']]
    X = df[['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw']]
    y = df[['target_x', 'target_y']]

    # Podział na dane treningowe i dane walidacyjne
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n--- Trenowanie modelu regresji ---")
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=0.1)
    )
    model.fit(X, y)

    # Zapis modelu do pliku
    joblib.dump(model, MODEL_FILE)
    print(f"Model zapisano do pliku: {MODEL_FILE}")

    """ Wizualizacja """
    if visualise:
        predictions = model.predict(X_val)

        # Obliczenia błędów
        mae_x = mean_absolute_error(y_val['target_x'], predictions[:, 0])
        mae_y = mean_absolute_error(y_val['target_y'], predictions[:, 1])

        mae_x_px = mae_x * SCREEN_WIDTH
        mae_y_px = mae_y * SCREEN_HEIGHT

        # Błąd euklidesowy dla każdej próbki
        errors = np.sqrt(((predictions[:, 0] - y_val['target_x']) * SCREEN_WIDTH) ** 2 +
                         ((predictions[:, 1] - y_val['target_y']) * SCREEN_HEIGHT) ** 2)
        mean_error_px = np.mean(errors)

        print("\n" + "=" * 40)
        print(f" WYNIKI EWALUACJI")
        print("=" * 40)
        print(f"Średni błąd X: {mae_x_px:.1f} px")
        print(f"Średni błąd Y: {mae_y_px:.1f} px")
        print(f"ŚREDNI BŁĄD CAŁKOWITY (odległość): {mean_error_px:.1f} px")
        print("=" * 40)

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