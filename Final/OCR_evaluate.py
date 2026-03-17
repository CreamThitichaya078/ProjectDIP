import re

# ฟังก์ชันคำนวณ Levenshtein Distance (จำนวนครั้งที่ต้องแก้ไขให้ข้อความเหมือนกัน)
def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))  # สร้าง array สำหรับเก็บค่าระยะทาง

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev  # ถ้าตัวอักษรเหมือนกัน ไม่ต้องแก้
            else:
                # เลือกค่าที่น้อยที่สุดระหว่าง insert, delete, replace
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n]  # ค่าระยะทางสุดท้าย


# ปรับรูปแบบข้อความก่อนนำไปคำนวณ
def normalize(text):
    text = text.lower()  # แปลงเป็นตัวพิมพ์เล็กทั้งหมด
    text = re.sub(r'\s+', ' ', text)  # รวมช่องว่างหลายตัวให้เหลือช่องว่างเดียว
    return text.strip()  # ลบช่องว่างหัวท้าย


# คำนวณค่า CER และ WER
def calculate_metrics(gt, hyp):
    # --- CER (Character Error Rate) ---
    gt_n, hyp_n = normalize(gt), normalize(hyp)
    dist_c = levenshtein(gt_n, hyp_n)  # ระยะห่างระดับตัวอักษร
    cer_score = dist_c / max(len(gt_n), 1)  # ป้องกันหารด้วย 0

    # --- WER (Word Error Rate) ---
    gt_w, hyp_w = gt_n.split(), hyp_n.split()  # แยกเป็นคำ
    dist_w = levenshtein(gt_w, hyp_w)  # ระยะห่างระดับคำ
    wer_score = dist_w / max(len(gt_w), 1)

    return cer_score, wer_score

## CER = จำนวนตัวอักษรที่ผิด / จำนวนตัวอักษรทั้งหมด
## WER = จำนวนคำที่ผิด / จำนวนคำทั้งหมด

# # รันภายใน
# if __name__ == '__main__':
#     calculate_metrics('', '')
