import cv2

cap = cv2.VideoCapture(0)

# Pobieranie konkretnych właściwości
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
backend = cap.get(cv2.CAP_PROP_BACKEND) # Sprawdza czy to MSMF, DSHOW itp.
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) # Kodek/Format (np. MJPG)

# Dekodowanie formatu FOURCC na czytelny tekst
codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"Rozdzielczość: {width}x{height}")
print(f"Klatki na sekundę (FPS): {fps}")
print(f"Format/Codec: {codec}")

# ret, frame = cap.read()
# print(frame.shape)

# Ustawienie szerokości (1280) i wysokości (720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if ret:
    print(f"Aktualna rozdzielczość: {frame.shape[1]}x{frame.shape[0]}")

cap.release()