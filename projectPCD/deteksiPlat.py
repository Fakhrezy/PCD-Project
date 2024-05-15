import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load citra RGB (BGR)
img = cv.imread('project/data/plat.jpg')

# Resize citra dengan mengalikannya ukuran aslinya dengan 0.4
img = cv.resize(img, (int(img.shape[1]*.4),int(img.shape[0]*.4)))

# Normalisasi Cahaya
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
img_opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
img_norm = img - img_opening

# Konversi citra keabuan (grayscale)
img_gray = cv.cvtColor(img_norm, cv.COLOR_BGR2GRAY)

# Thresholding menggunakan metode Otsu
_, img_thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Deteksi plat menggunakan contours
contours_vehicle, _ = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Cek jumlah contours
index_plate_candidate = []

for contour_vehicle in contours_vehicle:
    x,y,w,h = cv.boundingRect(contour_vehicle)
    aspect_ratio = w/h
    if w >= 200 and aspect_ratio <= 4 : 
        index_plate_candidate.append(contour_vehicle)

if len(index_plate_candidate) == 0:
    print("Plat nomor tidak ditemukan")
else:
    plate_candidate = max(index_plate_candidate, key=cv.contourArea)
    x_plate, y_plate, w_plate, h_plate = cv.boundingRect(plate_candidate)
    cv.rectangle(img, (x_plate, y_plate), (x_plate+w_plate, y_plate+h_plate), (0,255,0), 5)
    plate_roi = img[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]

    # Konversi ke citra keabuan (grayscale)
    plate_gray = cv.cvtColor(plate_roi, cv.COLOR_BGR2GRAY)

    # Thresholding menggunakan metode Otsu
    _, img_plate_thresh = cv.threshold(plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Segmentasi karakter menggunakan contours
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    img_plate_thresh = cv.morphologyEx(img_plate_thresh, cv.MORPH_OPEN, kernel)
    contours_plate, _ = cv.findContours(img_plate_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

    index_chars_candidate = []

    for contour_plate in contours_plate:
        x_char,y_char,w_char,h_char = cv.boundingRect(contour_plate)
        if h_char >= 40 and h_char <= 60 and w_char >= 10:
            index_chars_candidate.append(contour_plate)

    if index_chars_candidate == []:
        print('Karakter tidak tersegmentasi')
    else:
        for char_candidate in index_chars_candidate:
            x,y,w,h = cv.boundingRect(char_candidate)
            cv.rectangle(img, (x_plate+x, y_plate+y), (x_plate+x+w, y_plate+y+h), (0,255,0), 5)

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
