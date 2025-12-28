import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error
from pyautogui import size


TEST_FILE = "data/test_data.csv"  # Plik z sesji testowej
MODEL_FILE = "data/calibration_model.pkl"  # Plik trenowanego modelu
SCREEN_WIDTH, SCREEN_HEIGHT = size()


def evaluate():
    if not os.path.exists(MODEL_FILE):
        print(f"BŁĄD: Nie znaleziono modelu {MODEL_FILE}. Najpierw uruchom trenowanie")
        return
    if not os.path.exists(TEST_FILE):
        print(f"BŁĄD: Nie znaleziono pliku {TEST_FILE}. Uruchom WMediapipe.py w trybie 2.")
        return

    print(f"--- Wczytywanie modelu i danych testowych ---")
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(TEST_FILE)

    X_test = df[['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll', 'eye_aspect_ratio']]
    y_true = df[['target_x', 'target_y']]

    print(f"Liczba próbek testowych: {len(df)}")

    # Predykcja
    predictions = model.predict(X_test)

    # Obliczenia błędów
    mae_x = mean_absolute_error(y_true['target_x'], predictions[:, 0])
    mae_y = mean_absolute_error(y_true['target_y'], predictions[:, 1])

    mae_x_px = mae_x * SCREEN_WIDTH
    mae_y_px = mae_y * SCREEN_HEIGHT

    # Błąd euklidesowy dla każdej próbki
    errors = np.sqrt(((predictions[:, 0] - y_true['target_x']) * SCREEN_WIDTH) ** 2 +
                     ((predictions[:, 1] - y_true['target_y']) * SCREEN_HEIGHT) ** 2)
    mean_error_px = np.mean(errors)

    print("\n" + "=" * 40)
    print(f" WYNIKI EWALUACJI")
    print("=" * 40)
    print(f"Średni błąd X: {mae_x_px:.1f} px")
    print(f"Średni błąd Y: {mae_y_px:.1f} px")
    print(f"ŚREDNI BŁĄD CAŁKOWITY (odległość): {mean_error_px:.1f} px")
    print("=" * 40)

    """Wizualizacja"""
    plt.figure(figsize=(12, 8))
    # Rysowanie predykcji
    plt.scatter(predictions[:, 0], predictions[:, 1], c=y_true['target_x'], cmap='viridis', alpha=0.3, s=10,
                label='Estymacja')

    for i in range(len(predictions)):
       plt.plot([y_true.iloc[i]['target_x'], predictions[i,0]], [y_true.iloc[i]['target_y'], predictions[i,1]], 'gray', alpha=0.1)

    # Rysowanie środków celów
    unique_targets = df[['target_x', 'target_y']].drop_duplicates().values
    plt.scatter(unique_targets[:, 0], unique_targets[:, 1], c='red', marker='+', s=200, linewidth=3, label='Cel')

    plt.title(f"Rozrzut spojrzenia dla danych testowych\nŚredni błąd: {mean_error_px:.1f} px")
    plt.xlabel("Ekran X")
    plt.ylabel("Ekran Y")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    evaluate()