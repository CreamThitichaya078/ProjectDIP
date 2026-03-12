import cv2
import numpy as np
import pywt

og_receipt = cv2.imread('D:/DIP/Project/OCR/Test/Poundland_Up.JPG', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('og_receipt', og_receipt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------
#     แยกใบเสร็จกับพื้นหลัง
#-----------------------------
otsu_val, mask = cv2.threshold(og_receipt, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU) #หาค่าอัตโนมัติ
# cv2.imshow('Otsu', mask )
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(f"Otsu Threshold = {otsu_val}") #val = 107.0 น้อยกว่า=พื้นหลัง/มากกว่า=ใบเสร็จ

kernel = np.ones((5,5), np.uint8)
mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel, iterations=3) #รอยพับทำให้ข้อความบางส่วนหาย ใช้ MORPH_CLOSE เติมจุดที่่หายไปของภาพ
# cv2.imshow('Morphological Close', mask2 )
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours, _ = cv2.findContours(mask2,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cv2.RETR_EXTERNAL เอาแค่ขอบนอก / cv2.CHAIN_APPROX_SIMPLE บีบอัดจุด
largest = max(contours, key=cv2.contourArea) #หา Contour ใหญ่สุด = ใบเสร็จ
#ยังไม่เก้ท
final_mask = np.zeros_like(og_receipt)
cv2.drawContours(final_mask,[largest], -1, 255, -1  )
result = cv2.bitwise_and(og_receipt, og_receipt,mask=final_mask)
#ยังไม่เก้ท
# cv2.imshow("final_mask", final_mask)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------
#   Perspective Transform ให้ภาพตรงขึ้น
#-----------------------------------------
#ยังไม่เก้ท
epsilon = 0.02 * cv2.arcLength(largest, True) #หามุมของใบเสร็จ ต้องได้ 4 มุม
approx = cv2.approxPolyDP(largest, epsilon, True) #ลดจุดใน Contour
#ยังไม่เก้ท
# print(f"จำนวนมุมที่หาได้ = {len(approx)}") #ได้ 4 มุม

pts = approx.reshape(4, 2).astype(np.float32) #เรียงมุมให้ถูกต้อง เพราะ findContours คืนมุมแบบสุ่ม
rect = np.zeros((4, 2), dtype=np.float32)
s = pts.sum(axis=1)
rect[0]  = pts[np.argmin(s)]   # บนซ้าย = ผลรวม x+y น้อยสุด
rect[2]  = pts[np.argmax(s)]   # ล่างขวา = ผลรวม x+y มากสุด
diff = np.diff(pts, axis=1)
rect[1] = pts[np.argmin(diff)] # บนขวา = ผลต่าง y-x น้อยสุด
rect[3] = pts[np.argmax(diff)] # ล่างซ้าย = ผลต่าง y-x มากสุด
#คำนวณขนาด เพราะใบเสร็จแต่ละใบขนาดต่างกัน
widthA = np.linalg.norm(rect[2] - rect[3]) #ความกว้าง = ระยะห่างขอบบน และ ขอบล่าง
widthB = np.linalg.norm(rect[1] - rect[0])
heightA = np.linalg.norm(rect[1] - rect[2])  #ความสูง = ระยะห่างขอบซ้าย และ ขอบขวา
heightB = np.linalg.norm(rect[0] - rect[3])
maxW = int(max(widthA, widthB))  #เอาค่ามากสุดเพื่อไม่ให้ภาพเสียข้อมูล
maxH = int(max(heightA, heightB))

# print(f"ขนาด Output = {maxW} x {maxH}") #Output = 496 x 1171

#จุดปลายทาง
dst = np.array([
        [0,       0], #บนซ้าย
        [maxW,    0], #บนขวา
        [maxW, maxH], #ล่างขวา
        [0,    maxH] #ล่างซ้าย
    ], dtype=np.float32)

M = cv2.getPerspectiveTransform(rect, dst)  #คำนวณ Perspective Matrix
warped = cv2.warpPerspective(result, M, (maxW, maxH)) #Transform
# cv2.imshow("warped", warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------------------------
#      Wavelet Sharpen เพิ่มความคมชัดขอบตัวอักษร
#-----------------------------------------------
img_float = warped.astype(np.float32) / 255.0 #pywt.wavedec2 ต้องการค่า Float และแม่นยำกว่า
coeffs = pywt.wavedec2(img_float, wavelet= 'db4', level= 3) #Decompose แยกภาพออกเป็นความถี่ต่างๆ
#ขยาย Detail Coefficients
#ขอบที่มีอยู่แล้ว → คมขึ้น / พื้นที่เรียบ → ไม่เปลี่ยน
new_coeffs = [coeffs[0]]  # LL ไม่แตะ
for detail in coeffs[1:]:
    new_coeffs.append(tuple(c * 1.5 for c in detail))
# Reconstruct
sharp = pywt.waverec2(new_coeffs, 'db4') #รวม Coefficients กลับเป็นภาพ
sharp = np.clip(sharp * 255, 0, 255).astype(np.uint8) #ตัดค่าที่เกิน 255 หรือต่ำกว่า 0
warped = sharp
# cv2.imshow('Sharp', warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-----------------------------
#           ปรับแสง
#-----------------------------
warped = cv2.GaussianBlur(warped, (5, 5), 0) #ลด noise ลายน้ำ
# cv2.imshow('denoise', warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (16,16)) #จำกัดการขยาย Contrast และ Grid เพื่อให้ภาพเป็นธรรมชาติข้อความสว่างสม่ำเสมอ
cl = clahe.apply(warped)

cv2.imwrite('output_final_Up_Chompoo.jpg', cl)
print('เสร็จแล้ว บันทึกที่ output_final_Up_Chompoo.jpg')
# cv2.imshow('clahe', cl)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# thresh = cv2.adaptiveThreshold(cl, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15, 8)
# kernel = np.ones((2,2), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations=1) #Morphological Opening ลบ Noise จุดเล็กๆ
# kernel_close = np.ones((1, 1), np.uint8)
# thresh = cv2.medianBlur(thresh, 3) # ลบ Noise จุดดำในพื้นขาว
# cv2.imshow('thresh',thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
