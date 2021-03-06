# Projekt zaliczeniowy WDSI
### Maciej Krupka

## Wymagane biblioteki
- import numpy
- import cv2
- import sklearn
- import pandas
- import xml.etree.ElementTree
- import pathlib 
- import os
- import matplotlib

## Lokalizacja znaków
W celu lokalizacji znaków skorzystałem z algorytmu **Hough'a**, tranformata w kazdym mozliwym punkcie wykrytej krawędzi obrazu tworzy środek okregu, wybierane są te które są lokalnymi maksimami. Za pomocą glosowania, zostały wybrane potencjalne punkty które sa kandydatami na środki okręgów, następnie w celu najlepszego dopasowania promienia, jest on wielokrotnie zwiększany. Podczas procesu wyszukiwane są najlepsze dopasowania okręgu do istniejących w obrazie. W celu uzyskania zadowalających wyników, obraz musiał zostać przetworzony do obrazu w odcieniach szarosci, następnie wygładzony za pomocą **filtru gausowskiego**

## Non maximum supression
Lokalizacja znaków zwróciła współrzędne prostokątów. Niektóre prostokąty zawierające znak pokrywały się ze sobą, a za pomocą techniki **Non maximum supression** wybierany jest obszar ktorego podobieństwo miedzy dwoma zbiorami jest największe, zwany **indeksem Jaccarda**. Omawiane zagadnienie zostało zaprezentowane ponizej: 
![Non maximum supression](https://github.com/macnack/projekt-zaliczeniowy/blob/master/images/nms.png)

## Mozliwosci ulepszenia
- Dodanie histogramu koloru do klasyfikiacji obrazow
- W celu zwiekszenia wydajnosci lokalizacji okręgów obraz wygładzony przetworzyć przez detekcje krawędzi **Canny Edge**
## Linki
Tutorial [Hough Circle Transform](https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html).

Tutorial [Non-Maximum Suppression](https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/).
