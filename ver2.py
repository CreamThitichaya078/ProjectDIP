import cv2
import numpy as np

# ============================================================
# โหลดภาพ
# ============================================================
img = cv2.imread('D:/DIP/Project/OCR/Test/Poundland_Up.JPG', cv2.IMREAD_GRAYSCALE)

# ============================================================
# Step 1 — ตัดพื้นหลังออก
# ============================================================
# Otsu หา threshold อัตโนมัติ พื้นดำ=0 ใบเสร็จ=255
_, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological Close เติมรอยขาดที่เกิดจากรอยพับ
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

# หา contour ที่ใหญ่สุด = ใบเสร็จ
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

# วาด mask ทึบครอบใบเสร็จ แล้ว AND กับภาพต้นฉบับ
final_mask = np.zeros_like(img)
cv2.drawContours(final_mask, [largest], -1, 255, -1)
masked = cv2.bitwise_and(img, img, mask=final_mask)

# ============================================================
# Step 2 — Crop เหลือแค่ใบเสร็จ
# ============================================================

x, y, w, h = cv2.boundingRect(largest)
cropped = masked[y:y+h, x:x+w]

# ============================================================
# Step 3 — Background Division แก้ bleed-through
# ============================================================

# Gaussian blur ขนาดใหญ่มาก = ประมาณค่าแสงพื้นหลัง
# ตัวอักษรจะหายไปหมด เหลือแต่ gradient ของแสง
bg = cv2.GaussianBlur(cropped.astype(np.float32), (91, 91), 0)

# หาร: พื้นหลัง/พื้นหลัง ≈ 1.0 (ขาว), ตัวอักษร/พื้นหลัง << 1.0 (เข้ม)
divided = cropped.astype(np.float32) / (bg + 1e-6)

# normalize กลับมาเป็น 0-255
divided = cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ============================================================
# Step 4 — Gaussian Blur ลด noise เบาๆ
# ============================================================

blurred = cv2.GaussianBlur(divided, (3, 3), 0)

# ============================================================
# Step 5 — CLAHE เพิ่ม contrast
# ============================================================

# clipLimit=3.0 จำกัดการขยาย contrast ไม่ให้ noise ถูก amplify มากเกิน
# tileGridSize=(8,8) แบ่งภาพเป็น grid ปรับ contrast แยกแต่ละบริเวณ
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
cl = clahe.apply(blurred)

# ============================================================
# Step 6 — Adaptive Threshold แปลงเป็น Binary
# ============================================================

# block_size=51 ดูบริเวณกว้างขึ้นในการคำนวณ threshold
# C=10 ลบออกจาก mean เพื่อให้ตัวอักษรผ่าน threshold
# binary = cv2.adaptiveThreshold(
#     cl, 255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     21, 10
# )

# ============================================================
# Step 7 — Morphological ทำความสะอาด
# ============================================================

# ตัวอักษร=ดำ(0) พื้น=ขาว(255)
# Morphological ทำงานบน foreground=ขาว ต้อง invert ก่อน แล้ว invert กลับ
# se = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#
# inv    = cv2.bitwise_not(binary)              # ตัวอักษร=ขาว
# opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN,  se)   # ลบ noise จุดเล็กๆ
# closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se) # ซ่อมรอยขาดในตัวอักษร
# final  = cv2.bitwise_not(closed)              # ตัวอักษร=ดำ กลับมา

# ============================================================
# บันทึกผลลัพธ์
# ============================================================

cv2.imwrite('output_final_Up_Cream.jpg', cl)
print('เสร็จแล้ว บันทึกที่ output_final_Up_Cream.jpg')
