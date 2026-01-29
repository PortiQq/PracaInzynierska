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
