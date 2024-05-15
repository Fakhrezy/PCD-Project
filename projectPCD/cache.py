import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load gambar plat nomor
img = cv.imread('project/data/plathitam2.jpg')

# Prapengolahan
# Normalisasi Cahaya
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
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
    x, y, w, h = cv.boundingRect(contour_vehicle)
    aspect_ratio = w / h
    if w >= 200 and aspect_ratio <= 4:
        index_plate_candidate.append(contour_vehicle)

if len(index_plate_candidate) == 0:
    print("Plat nomor tidak ditemukan")
else:
    plate_candidate = max(index_plate_candidate, key=cv.contourArea)
    x_plate, y_plate, w_plate, h_plate = cv.boundingRect(plate_candidate)
    cv.rectangle(img, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), (0, 255, 0), 5)
    plate_roi = img[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]

    # Segmentasi Warna
    # Konversi ke model warna HSV
    hsv_plate = cv.cvtColor(plate_roi, cv.COLOR_BGR2HSV)

    # Tentukan rentang warna untuk hitam, putih, kuning, dan merah
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Buat mask untuk setiap warna
    mask_black = cv.inRange(hsv_plate, lower_black, upper_black)
    mask_white = cv.inRange(hsv_plate, lower_white, upper_white)
    mask_yellow = cv.inRange(hsv_plate, lower_yellow, upper_yellow)
    mask_red1 = cv.inRange(hsv_plate, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv_plate, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    # Klasifikasi Warna
    vehicle_classification = []

    if cv.countNonZero(mask_black) > 0:
        vehicle_classification.append("Pribadi (Hitam)")
    if cv.countNonZero(mask_white) > 0:
        vehicle_classification.append("Pribadi (Putih)")
    if cv.countNonZero(mask_yellow) > 0:
        vehicle_classification.append("Umum (Kuning)")
    if cv.countNonZero(mask_red) > 0:
        vehicle_classification.append("Pemerintah (Merah)")

    # Tampilkan hasil
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    # Tampilkan klasifikasi kendaraan berdasarkan warna plat nomor
    if vehicle_classification:
        most_common_classification = max(set(vehicle_classification), key=vehicle_classification.count)
        plt.text(10, 20, most_common_classification, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))
    else:
        plt.text(10, 20, "Plat nomor tidak terdeteksi", fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
