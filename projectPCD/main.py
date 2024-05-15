import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Fungsi untuk mengklasifikasikan warna kendaraan
def classify_color(mask_black, mask_white, mask_yellow, mask_red):
    # Hitung jumlah piksel warna pada setiap mask
    count_black = cv.countNonZero(mask_black)
    count_white = cv.countNonZero(mask_white)
    count_yellow = cv.countNonZero(mask_yellow)
    count_red = cv.countNonZero(mask_red)

    # Klasifikasi berdasarkan jumlah piksel warna
    if count_red > count_black and count_red > count_white and count_red > count_yellow:
        return "Kendaraan Pemerintah"
    elif count_black > count_white and count_black > count_yellow and count_black > count_red:
        return "Kendaraan Pribadi"
    elif count_white > count_black and count_white > count_yellow and count_white > count_red:
        return "Kendaraan Pribadi"
    elif count_yellow > count_black and count_yellow > count_white and count_yellow > count_red:
        return "Kendaraan Umum"
    else:
        return "Tidak Diketahui"

# Fungsi untuk melakukan segmentasi warna pada citra HSV
def segment_color(hsv_plate):
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

    return mask_black, mask_white, mask_yellow, mask_red

# Load gambar plat nomor
img = cv.imread('projectPCD/data/platmerah.jpg')

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

    # Crop gambar menggunakan koordinat plat nomor yang terdeteksi
    cropped_img = img[y_plate:y_plate + h_plate, x_plate:x_plate + w_plate]

    # Segmentasi Warna
    # Konversi ke model warna HSV
    hsv_plate = cv.cvtColor(cropped_img, cv.COLOR_BGR2HSV)

    # Lakukan segmentasi warna pada citra HSV
    mask_black, mask_white, mask_yellow, mask_red = segment_color(hsv_plate)

    # Klasifikasikan warna kendaraan
    classification = classify_color(mask_black, mask_white, mask_yellow, mask_red)
    
    # Tampilkan hasil
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    # Tampilkan klasifikasi kendaraan berdasarkan warna plat nomor
    plt.text(10, 20, classification, fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
