import cv2
import numpy as np
import mahotas
import pywt
import plotly.express as px

def preprocess(input_path, output_path):
    # To Grayscale

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # --- ส่วนที่ 1: การแยกวัตถุออกจากพื้นหลัง ---
    # 1.1 Otsu's Thresholding เพื่อแยกวัตถุ (ใบเสร็จ) ออกจากพื้นหลังโดยอัตโนมัติ

    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # T = mahotas.thresholding.otsu(img)
    # mask = np.where(img > T, 255, 0).astype(np.uint8)

    # 1.2 ใช้ Morphological Closing เพื่อเชื่อมจุดที่ขาดหายหรือรอยพับบนใบเสร็จให้สมบูรณ์ขึ้น
    kernel = np.ones((5, 5), np.uint8)
    for _ in range(3):
        mask = mahotas.close(mask, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # fig = px.imshow(mask, color_continuous_scale="gray")
    # fig.show()

    # 1.3 ค้นหาเส้นขอบ และเลือกขอบที่มีขนาดพื้นที่ใหญ่ที่สุดซึ่งคิดว่าเป็นใบเสร็จ
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)

    # 1.4 สร้าง Mask ทึบเฉพาะบริเวณใบเสร็จและนำไปซ้อนกับภาพเดิมเพื่อตัดส่วนเกินรอบนอกออก
    final_mask = np.zeros_like(img)
    cv2.drawContours(final_mask, [largest], -1, 255, -1)
    masked = cv2.bitwise_and(img, img, mask=final_mask)

    # --- ส่วนที่ 2: การจัดรูปทรงและการปรับปรุงคุณภาพภาพ (Image Enhancement) ---
    # 2.1 Crop ให้เหลือเฉพาะขอบเขตของใบเสร็จ
    x, y, w, h = cv2.boundingRect(largest)
    cropped = masked[y:y + h, x:x + w]

    # 2.2 ทำ Background Division เพื่อกำจัดแสงเงาที่ไม่สม่ำเสมอและปัญหาข้อความทะลุหลัง
        # 1. ทำ Gaussian Blur ขนาดใหญ่เพื่อประมาณค่าความสว่างของพื้นหลัง
    bg = cv2.GaussianBlur(cropped.astype(np.float32), (91, 91), 0)
    # bg = mahotas.gaussian_filter(cropped.astype(float), sigma=45)
        # 2. นำภาพมาหารด้วยพื้นหลังเพื่อปรับค่าความสว่างให้สม่ำเสมอทั่วทั้งภาพ
    divided = cropped.astype(np.float32) / (bg + 1e-6)
        # 3. ปรับค่าสีให้อยู่ใน 0-255
    divided = cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # แปลงภาพเป็น float
    img_float = divided.astype(np.float32) / 255.0

    # Wavelet decomposition
    coeffs = pywt.wavedec2(img_float, wavelet='db4', level=3)

    # LL = background (ไม่เปลี่ยน)
    new_coeffs = [coeffs[0]]

    # เพิ่ม detail coefficient เพื่อให้ตัวอักษรคมขึ้น
    for detail in coeffs[1:]:
        cH, cV, cD = detail
        cH = cH * 1.5
        cV = cV * 1.5
        cD = cD * 1.2  # diagonal เพิ่มน้อยกว่าเพื่อลด noise
        new_coeffs.append((cH, cV, cD))

    # reconstruct ภาพ
    sharp = pywt.waverec2(new_coeffs, 'db4')

    # ป้องกันค่าหลุดช่วง
    sharp = np.clip(sharp, 0, 1)

    # แปลงกลับเป็น uint8
    sharp = (sharp * 255).astype(np.uint8)

    warped = sharp

    # 2.3 ลด noise ขนาดเล็กด้วย Gaussian Blur บางๆ
    blurred = cv2.GaussianBlur(warped, (3, 3), 0)

    # 2.4 CLAHE เพิ่มความคมชัดของตัวอักษรด้วย (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    #5 16
    cl = clahe.apply(blurred)

    cv2.imwrite(output_path, cl)
    return output_path

# รันภายใน

if __name__ == '__main__':
    preprocess('Poundland_Up.JPG', 'test.jpg')
