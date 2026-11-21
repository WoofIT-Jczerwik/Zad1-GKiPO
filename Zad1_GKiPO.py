
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

# 1. Pobranie obrazu rastrowego z URL i wczytanie go do programu
url = "https://i1.jbzd.com.pl/contents/2026/11/normal/hjh1PVxpEvdd46tUh0j4qTe1FRmaeqwL.jpg"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers, timeout=15)
response.raise_for_status()

# Konwersja pobranych bajtów na tablicę NumPy i dekodowanie do macierzy obrazu (BGR)
image_array = np.frombuffer(response.content, dtype=np.uint8)
img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise ValueError("Nie udało się zdekodować obrazu z pobranych danych.")

# 2. Wyświetlenie oryginalnego obrazu za pomocą matplotlib (BGR -> RGB)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title("Pierwotny obraz")
plt.axis("off")
plt.show()

# 3. Zmniejszenie rozdzielczości o połowę oraz konwersja do skali szarości (grayscale)
height, width = img_bgr.shape[:2]
new_width = max(1, int(width * 0.5))
new_height = max(1, int(height * 0.5))

img_resized = cv2.resize(img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# 4. Obrót obrazu o 90 stopni (zgodnie z ruchem wskazówek zegara)
img_rotated = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

# 5. Wyświetlenie obrazu wynikowego
plt.figure(figsize=(6, 6))
plt.imshow(img_rotated, cmap="gray")
plt.title("Obraz po zmianie skali (do 50%), grayscale i obrocie o 90°")
plt.axis("off")
plt.show()

# 6. Wyświetlenie macierzy obrazu w postaci tablicy liczb
print("Kształt macierzy wynikowej (wysokość, szerokość):", img_rotated.shape)
print("Typ danych (dtype):", img_rotated.dtype)
print("Macierz obrazu (tablica liczb 0-255):")
print(img_rotated)
