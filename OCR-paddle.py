"""
PaddleOCR - Word Bounding Box Highlighter (v2.7.3)
===================================================
สิ่งที่ต้องติดตั้งก่อน (Terminal ใน PyCharm):
    pip uninstall paddlepaddle paddleocr paddlex -y
    pip install paddlepaddle==2.6.2
    pip install paddleocr==2.7.3 pillow
    pip install "numpy<2.0"

วิธีใช้:
    1. ใส่ชื่อรูปใน IMAGE_PATH
    2. รัน script นี้ใน PyCharm
"""

import os
# ปิด oneDNN เพื่อแก้ NotImplementedError บน Windows
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import colorsys
import random
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

# ============================================================
# ⚙️  CONFIG — แก้ตรงนี้เท่านั้น
# ============================================================
IMAGE_PATH   = "D:/DIP/Project/output_final_Up_Chompoo.jpg"  # ← ชื่อรูปของคุณ
OUTPUT_PATH  = "Result/output_OCR_final_Up_Cream.png"             # ← ชื่อไฟล์ผลลัพธ์
LANGUAGE     = "en"                                   # ← en=อังกฤษ, ch=จีน
ROTATE_ANGLE = 0                            # ← หมุนรูปก่อน OCR (0, 90, 180, 270)
# ============================================================

# ชื่อไฟล์ temp สำหรับรูปที่ rotate แล้ว
ROTATED_TEMP = "rotated_temp.jpg"


def get_random_color(alpha=160):
    """สุ่มสีสดใสแบบ HSV"""
    hue = random.random()
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), alpha)


def rotate_image(image_path: str, angle: int, temp_path: str) -> str:
    """หมุนรูปตามองศาที่กำหนด แล้วบันทึกเป็น temp file"""
    if angle == 0:
        return image_path  # ไม่ต้องหมุน

    img = Image.open(image_path)
    # expand=True ทำให้ canvas ขยายตามรูปที่หมุน
    rotated = img.rotate(angle, expand=True)
    rotated.save(temp_path)
    print(f"🔄 หมุนรูป {angle} องศา → บันทึกที่ {temp_path}")
    return temp_path


def run_ocr(image_path: str, lang: str):
    """รัน PaddleOCR และได้ word-level bounding boxes"""
    print("⏳ กำลังโหลด PaddleOCR model...")
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    result = ocr.ocr(image_path, cls=True)
    return result


def draw_word_boxes(image_path: str, result, output_path: str):
    """วาด colored bounding box รอบแต่ละคำ (รองรับ rotated box)"""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    word_count = 0

    if result is None:
        print("❌ ไม่พบข้อความในรูป")
        return 0

    for page in result:
        if page is None:
            continue
        for line in page:
            try:
                box, (text, conf) = line
                if not str(text).strip() or float(conf) < 0.3:
                    continue

                points = [(int(p[0]), int(p[1])) for p in box]
                color = get_random_color(alpha=170)
                draw.polygon(points, fill=color)
                draw.line(points + [points[0]], fill=color[:3] + (230,), width=2)
                word_count += 1

            except Exception as e:
                print(f"⚠️ ข้ามบรรทัดนี้เพราะ: {e}")
                continue

    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    combined = Image.alpha_composite(img, overlay)
    combined.convert("RGB").save(output_path)

    print(f"✅ OCR สำเร็จ! พบ {word_count} คำ")
    print(f"📁 บันทึกผลลัพธ์ที่: {output_path}")
    return word_count


def extract_text(result) -> str:
    """รวม text ทั้งหมดเป็น string"""
    lines = []
    if result is None:
        return ""
    for page in result:
        if page is None:
            continue
        for line in page:
            try:
                _, (text, conf) = line
                if str(text).strip() and float(conf) >= 0.3:
                    lines.append(str(text))
            except Exception:
                continue
    return "\n".join(lines)


def main():
    if not Path(IMAGE_PATH).exists():
        print(f"❌ ไม่พบไฟล์รูป: {IMAGE_PATH}")
        return

    print(f"🔍 กำลัง OCR ไฟล์: {IMAGE_PATH}  (ภาษา: {LANGUAGE})")

    # หมุนรูปก่อน OCR
    ocr_input = rotate_image(IMAGE_PATH, ROTATE_ANGLE, ROTATED_TEMP)

    # รัน OCR บนรูปที่ rotate แล้ว
    result = run_ocr(ocr_input, LANGUAGE)

    # วาด bounding boxes บนรูปที่ rotate แล้ว
    draw_word_boxes(ocr_input, result, OUTPUT_PATH)

    # แสดงและบันทึก text
    text = extract_text(result)
    print("\n📝 Text ที่ OCR ได้:")
    print("-" * 50)
    print(text[:500] + ("..." if len(text) > 500 else ""))

    txt_output = OUTPUT_PATH.replace(".png", "_text.txt")
    Path(txt_output).parent.mkdir(parents=True, exist_ok=True)
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n📄 บันทึก text ที่: {txt_output}")

    # ลบ temp file
    if ROTATE_ANGLE != 0 and Path(ROTATED_TEMP).exists():
        Path(ROTATED_TEMP).unlink()


if __name__ == "__main__":
    main()