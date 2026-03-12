import os
# ปิด oneDNN เพื่อแก้ NotImplementedError บน Windows
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import colorsys
import random
from pathlib import Path
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw

def get_random_color(alpha=160):
    hue = random.random()
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255), alpha)

def rotate_image(image_path, angle, temp_path):
    if angle == 0: return image_path
    img = Image.open(image_path)
    rotated = img.rotate(angle, expand=True)
    rotated.save(temp_path)
    return temp_path

def run_ocr(image_path, lang):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return ocr.ocr(image_path, cls=True)

def draw_word_boxes(image_path, result, output_path):
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if result and result[0]:
        for line in result[0]:
            box, (text, conf) = line
            points = [(int(p[0]), int(p[1])) for p in box]
            color = get_random_color()
            draw.polygon(points, fill=color)
            draw.line(points + [points[0]], fill=color[:3] + (230,), width=2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img, overlay).convert("RGB").save(output_path)

def extract_text(result):
    lines = []
    if result and result[0]:
        for line in result[0]:
            _, (text, conf) = line
            if float(conf) >= 0.3: lines.append(text)
    return "\n".join(lines)

def run_ocr_pipeline(image_path, output_path, rotate_angle=0, lang="en"):
    temp_rotated = "temp_rotate.jpg"
    ocr_input = rotate_image(image_path, rotate_angle, temp_rotated)
    result = run_ocr(ocr_input, lang)
    draw_word_boxes(ocr_input, result, output_path)
    text = extract_text(result)

    txt_path = Path(output_path).with_suffix("").as_posix() + "_text.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    if rotate_angle != 0 and os.path.exists(temp_rotated): os.remove(temp_rotated)
    return text

# รันภายใน
# if __name__ == '__main__':
#     run_ocr_pipeline('', '')