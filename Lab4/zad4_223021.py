import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use as mpl_use
from pathlib import Path
mpl_use('TkAgg')

# TUKA GO KORISTAM ISTIOT ALGORITAM ZA DOBIVANJE KONTURI OD MINATATA DOMASNA NAPRAVEN VO FUNKCIJA
def contours_image(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret, thresholdedImg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(thresholdedImg, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    contoursOpened, hierarchy1 = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursOpenedImage = cv2.drawContours(np.zeros_like(image), contoursOpened, -1, (255, 255, 255), 1)
    return contoursOpenedImage

# TUKA SOODVETNO STAVETE JA PATEKATA ZA QUERY IMAGE-OT
val = input("Vnesete ime na slika od query folderot vo format ime.jpg (primer 14147.jpg): ")
queryImage = cv2.imread('./query_images/' + val, 0)
# queryImage = cv2.imread('./query_images/14147.jpg', 0)
contoursQueryImage = contours_image(queryImage)
similarities = {}

# TUKA SOODVETNO STAVETE JA PATEKATA ZA DATABASE FOLDER-OT
inputFolder = Path('./database/')
images = list(inputFolder.glob('*'))

for image in images:
    img = cv2.imread(str(image), 0)
    contoursImg = contours_image(img)
    similarities[image.name] = cv2.matchShapes(contoursQueryImage, contoursImg, 1, 0)

for key, value in sorted(similarities.items(), key=lambda item: item[1]):
    print(key, value)

# Generalno rezultatite se takvi kako sto gi ocekuvav, za listovite so slicna forma funkcijata matchShapes vrakja mal broj kako na primer
# pri sporedba na 14147.jpg (query image-ot) so 11628.jpg (database image) vrakja 0.034, dodeka pak pri spredba so sliki od listovi so sosema razlicna forma kako na primer
# 1.jpg, funckijata vrakja pogolem broj vo slucajov 0.439.
# Dopolnitelno postaviv i slika od rezultatite vo folderot.

