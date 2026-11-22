import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


def pobierz_obraz_zdalny(url: str, user_agent: str = "Mozilla/5.0") -> np.ndarray:
    # Funkcja pobiera obraz ze zdalnego serwera i dekoduje go do macierzy BGR.
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    image_array = np.frombuffer(response.content, dtype=np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Nie udalo sie zdekodowac obrazu z pobranych danych.")

    return img_bgr


def wyswietl_obraz(obraz: np.ndarray, tytul: str, czy_bgr: bool = False, czy_szary: bool = False) -> None:
    # Funkcja Wyswietla obraz 
    obraz_do_pokazania = obraz
    if czy_bgr:
        obraz_do_pokazania = cv2.cvtColor(obraz, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    if czy_szary:
        plt.imshow(obraz_do_pokazania, cmap="gray")
    else:
        plt.imshow(obraz_do_pokazania)
    plt.title(tytul)
    plt.axis("off")
    plt.show()


def przetworz_obraz(obraz_bgr: np.ndarray, skala: float = 0.5) -> np.ndarray:
    # Funkcja zmniejsza rozdzielczosc obrazu, konwertuje do skali szarosci i obraca o 90 stopni.
    height, width = obraz_bgr.shape[:2]
    new_width = max(1, int(width * skala))
    new_height = max(1, int(height * skala))

    img_resized = cv2.resize(obraz_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_rotated = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

    return img_rotated


def wyswietl_macierz(macierz: np.ndarray) -> None:
    # Funkcja wyswietla informacje o macierzy obrazu oraz jej zawartosc w postaci tablicy liczb
    print("Ksztalt macierzy wynikowej (wysokosc, szerokosc):", macierz.shape)
    print("Typ danych (dtype):", macierz.dtype)
    print("Macierz obrazu (tablica liczb 0-255):")
    print(macierz)


def main() -> None:
    url = "https://i1.jbzd.com.pl/contents/2026/11/normal/hjh1PVxpEvdd46tUh0j4qTe1FRmaeqwL.jpg"

    img_bgr = pobierz_obraz_zdalny(url)
    wyswietl_obraz(img_bgr, "Oryginalny obraz zdalny", czy_bgr=True)

    img_wynikowy = przetworz_obraz(img_bgr, skala=0.5)
    wyswietl_obraz(img_wynikowy, "Obraz po zmianie skali 50%, grayscale i obrocie o 90 stopni", czy_szary=True)

    wyswietl_macierz(img_wynikowy)


if __name__ == "__main__":
    main()