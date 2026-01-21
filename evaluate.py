import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyautogui import size

# --- KONFIGURACJA ---
INPUT_FILE = "data/validation_data.csv"
SCREEN_WIDTH, SCREEN_HEIGHT = size()
SCREEN_WIDTH_MM = 345  # Np. 345 mm dla typowego laptopa 15.6 cala
USER_DISTANCE_MM = 400 # Przyjmujemy standardowe 60 cm, jeśli nie masz danych z czujnika

def pixels_to_degrees(error_px, screen_width_mm=SCREEN_WIDTH_MM, distance_mm=USER_DISTANCE_MM):
    """
    Konwertuje błąd z pikseli na stopnie kąta widzenia.
    """
    mm_per_pixel = screen_width_mm / SCREEN_WIDTH
    error_mm = error_px * mm_per_pixel
    angle = np.degrees(np.arctan(error_mm / distance_mm))

    return angle

def evaluate_session():
    if not os.path.exists(INPUT_FILE):
        print(f"BŁĄD: Nie znaleziono pliku {INPUT_FILE}. Uruchom najpierw walidację (klawisz 'T').")
        return

    print(f"--- Analiza pliku: {INPUT_FILE} ---")
    df = pd.read_csv(INPUT_FILE)

    if len(df) == 0:
        print("Plik jest pusty!")
        return

    # Obliczenie błędów dla każdego punktu
    df['error_x'] = np.abs(df['point_x'] - df['screen_x'])
    df['error_y'] = np.abs(df['point_y'] - df['screen_y'])
    df['error_dist'] = np.sqrt(df['error_x'] ** 2 + df['error_y'] ** 2)

    # --- STATYSTYKI ---
    mae_x = df['error_x'].mean()
    mae_y = df['error_y'].mean()
    mean_error = df['error_dist'].mean()
    max_error = df['error_dist'].max()

    mean_error_deg = pixels_to_degrees(mean_error)

    print("\n" + "=" * 40)
    print(f" WYNIKI WALIDACJI (Liczba próbek: {len(df)})")
    print("=" * 40)
    print(f"Średni błąd X: {mae_x:.1f} px")
    print(f"Średni błąd Y: {mae_y:.1f} px")
    print(f"ŚREDNI BŁĄD CAŁKOWITY (Mean):   {mean_error:.1f} px")
    print(f"ŚREDNI BŁĄD CAŁKOWITY w stopniach (Mean):   {mean_error_deg:.1f} STOPNI")
    print(f"Maksymalny błąd: {max_error:.1f} px")
    print("=" * 40)

    # --- WIZUALIZACJA ---
    plt.figure(figsize=(12, 8))

    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.axhline(SCREEN_HEIGHT, color='black', linewidth=1)
    plt.axvline(SCREEN_WIDTH, color='black', linewidth=1)

    # Rysowanie wszystkich predykcji
    scatter = plt.scatter(df['screen_x'], df['screen_y'],
                          c=df['error_dist'], cmap='coolwarm',
                          alpha=0.6, s=30, label='Pozycja Kursora')

    plt.colorbar(scatter, label='Błąd odległości (px)')

    # Rysowanie celów
    unique_targets = df[['point_x', 'point_y']].drop_duplicates()
    plt.scatter(unique_targets['point_x'], unique_targets['point_y'],
                c='lime', marker='X', s=200, edgecolors='black', linewidth=2, label='Cel (Target)')

    # Linie łączące
    for i in range(len(df)):
        plt.plot([df.iloc[i]['point_x'], df.iloc[i]['screen_x']],
                 [df.iloc[i]['point_y'], df.iloc[i]['screen_y']],
                 color='gray', alpha=0.1)

    plt.title(f"Wyniki Walidacji (Mean Error: {mean_error:.1f} px)\n"
              f"Zielony X = Cel, Kropki = Kursor")
    plt.xlabel("Ekran X (px)")
    plt.ylabel("Ekran Y (px)")

    # Ustawienie zakresu wykresu na wielkość ekranu (z lekkim marginesem)
    plt.xlim(-50, SCREEN_WIDTH + 50)
    plt.ylim(-50, SCREEN_HEIGHT + 50)
    plt.gca().invert_yaxis()  # Odwrócenie osi Y
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_session()