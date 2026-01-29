# Repozytorium pracy inżynierskiej - Inteligentny system pomiarów optometrycznych
## Opis projektu:
System śledzący wzrok użytkownika i mapujący go na współrzędne ekranu w czasie rzeczywistym. Użycie przy pomocy dowolnej kamery internetowej. Śledzenie położenia źrenic w obrębie oka oraz ułożenia głowy. 

## Technologie:
- Python
- OpenCV
- MediaPipe

## Wymagania systemowe

### Sprzęt
- **Komputer:** System operacyjny Windows, Linux lub macOS.
- **Kamera:** Kamera internetowa (wbudowana lub zewnętrzna) podłączona do komputera.

### Oprogramowanie
- **Python:** Interpreter w wersji 3.8 lub nowszej.

Wymagane zależności umieszczono w pliku requirements.txt. 

## Instrukcja uruchomienia projektu: 

### Pobranie projektu
Skopiowanie plików źródłowych projektu do wybranego katalogu na dysku lokalnym lub klonowanie repozytorium:
```bash
git clone https://github.com/PortiQq/PracaInzynierska
cd <Katalog projektu>
```

### Konfiguracja środowiska
Zaleca się utworzenie wirtualnego środowiska Python, aby odizolować zależności projektu:
```bash
python -m venv venv
```
- Windows:
```bash
venv\Scripts\activate
```
- Linux/macOS:
```bash
source venv/bin/activate
```
### Instalacja zależności:
```bash
pip install -r requirements.txt
```
### Uruchomienie:
- Program główny:
```bash
python main.py 
```
- Badanie skuteczności:
```bash
python evaluate.py
```

### Przykładowe dane
W katalogu `data/sample_data` znajdują się przykładowe pliki z danymi rejestrowanymi przez system:

| Nazwa pliku | Opis zawartości |
|:---|:---|
| `calibration_data.csv` | Dane rejestrowane podczas kalibracji (pozycja głowy/oczu ↔ punkt na ekranie), służące do trenowania modelu. |
| `validation_data.csv` | Zbiór danych testowych wykorzystywany do ewaluacji dokładności i precyzji systemu. |
| `session_data.csv` | Dane zebrane w trakcie swobodnej sesji użytkowania programu (już po procesie kalibracji). |
