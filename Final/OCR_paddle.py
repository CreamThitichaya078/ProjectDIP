import os
# ปิด oneDNN เพื่อแก้ปัญหา NotImplementedError บน Windows
os.environ["FLAGS_use_mkldnn"] = "0"
# ปิดการตรวจสอบแหล่งที่มาของโมเดล (ลด warning บางกรณี)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import colorsys
import random
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

# สุ่มสี (RGBA) สำหรับใช้วาดกรอบข้อความ
def get_random_color(alpha=160):
    hue = random.random()  # สุ่มค่า hue
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)  # แปลงเป็น RGB
    return (int(r * 255), int(g * 255), int(b * 255), alpha)

# หมุนภาพตามองศาที่กำหนด (ถ้า angle = 0 จะไม่หมุน)
def rotate_image(image_path, angle, temp_path):
    if angle == 0: return image_path
    img = Image.open(image_path)
    rotated = img.rotate(angle, expand=True)  # expand=True เพื่อไม่ให้ภาพถูกตัด
    rotated.save(temp_path)  # บันทึกเป็นไฟล์ชั่วคราว
    return temp_path

# เรียกใช้งาน PaddleOCR เพื่ออ่านข้อความจากภาพ
def run_ocr(image_path, lang):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return ocr.ocr(image_path, cls=True)  # cls=True = ใช้ angle classification

# วาดกรอบล้อมรอบคำที่ตรวจจับได้ แล้วบันทึกภาพ
def draw_word_boxes(image_path, result, output_path):
    img = Image.open(image_path).convert("RGBA")  # แปลงเป็น RGBA เพื่อรองรับ transparency
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))  # layer โปร่งใสสำหรับวาด
    draw = ImageDraw.Draw(overlay)

    if result and result[0]:
        for line in result[0]:
            box, (text, conf) = line  # box = พิกัด, text = ข้อความ, conf = confidence
            points = [(int(p[0]), int(p[1])) for p in box]  # แปลงพิกัดเป็น int
            color = get_random_color()  # สุ่มสี
            draw.polygon(points, fill=color)  # ระบายกรอบ
            draw.line(points + [points[0]], fill=color[:3] + (230,), width=2)  # วาดขอบ

    # สร้างโฟลเดอร์ (ถ้ายังไม่มี)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # รวมภาพต้นฉบับกับ overlay แล้วบันทึก
    Image.alpha_composite(img, overlay).convert("RGB").save(output_path)

# ดึงเฉพาะข้อความจากผล OCR (กรองตาม confidence)
def extract_text(result):
    lines = []
    if result and result[0]:
        for line in result[0]:
            _, (text, conf) = line
            if float(conf) >= 0.3:  # เก็บเฉพาะข้อความที่มั่นใจ >= 0.3
                lines.append(text)
    return "\n".join(lines)  # รวมเป็น string หลายบรรทัด

# pipeline หลัก: หมุนภาพ → OCR → วาดกรอบ → ดึงข้อความ → บันทึก text
def run_ocr_pipeline(image_path, output_path, rotate_angle=0, lang="en"):
    temp_rotated = "temp_rotate.jpg"

    # 1. หมุนภาพ (ถ้ามี)
    ocr_input = rotate_image(image_path, rotate_angle, temp_rotated)

    # 2. ทำ OCR
    result = run_ocr(ocr_input, lang)

    # 3. วาด bounding box ลงบนภาพ
    draw_word_boxes(ocr_input, result, output_path)

    # 4. ดึงข้อความ
    text = extract_text(result)

    # 5. บันทึกข้อความเป็นไฟล์ .txt
    txt_path = Path(output_path).with_suffix("").as_posix() + "_text.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    # 6. ลบไฟล์ชั่วคราว (ถ้ามีการหมุน)
    if rotate_angle != 0 and os.path.exists(temp_rotated):
        os.remove(temp_rotated)

    return text

# รันภายใน (สำหรับทดสอบไฟล์นี้โดยตรง)
# if __name__ == '__main__':
#     run_ocr_pipeline('', '')
