from preprocess import preprocess
from OCR_Paddle import run_ocr_pipeline
from OCR_Evaluate import calculate_metrics
import plotly.express as px
import cv2

# --- CONFIG PATH  ---
INPUT_IMAGE = "D:/DIP/Project/Final/Poundland.jpg" #ภาพต้นฉบับ
PREPROCESSED_IMG = "D:/DIP/Project/Final/output_Pre_Poundland.jpg" #ภาพต้นฉบับที่ผ่าน image processing แล้ว
OCR_OUTPUT_PRE = "D:/DIP/Project/Final/output_Pre_OCR_Poundland.jpg" #ภาพที่ผ่าน image processing แล้ว ไปเข้า OCRต่อ
OCR_OUTPUT_ORI = "D:/DIP/Project/Final/output_Ori_OCR_Poundland.jpg" #ภาพต้นฉบับที่ผ่าน OCR
GROUND_TRUTH_TXT = "D:/DIP/Project/Final/ground_truth.txt" #ผลเฉลย


def main():
    # อ่านไฟล์ groud truth
    with open(GROUND_TRUTH_TXT, 'r', encoding='utf-8') as f:
        gt_text = f.read()

    # 1. ทำ Image Preprocessing
    print("\n--- [1/3] Processing Image ---")
    preprocess(INPUT_IMAGE, PREPROCESSED_IMG)

    # 2. ทำ OCR ภาพ
    print("\n--- [2/3] OCR ---")
    text_ori = run_ocr_pipeline(INPUT_IMAGE, OCR_OUTPUT_ORI,270)
    text_pre = run_ocr_pipeline(PREPROCESSED_IMG, OCR_OUTPUT_PRE)

    # 3. สรุปผลการเปรียบเทียบ
    print("\n--- [3/3] Evaluate ---")
    cer_pre, wer_pre = calculate_metrics(gt_text, text_pre)
    cer_ori, wer_ori = calculate_metrics(gt_text, text_ori)

    print("=" * 50)
    print(f"{'Metric':<15} | {'Original':<15} | {'Preprocessed':<15}")
    print("-" * 50)
    print(f"{'CER':<15} | {cer_ori * 100:>13.2f}% | {cer_pre * 100:>13.2f}%")
    print(f"{'WER':<15} | {wer_ori * 100:>13.2f}% | {wer_pre * 100:>13.2f}%")
    print("=" * 50)

    # 4. แสดงภาพเปรียบเทียบ (Option)
    fig = px.imshow(cv2.imread(INPUT_IMAGE))
    fig2 = px.imshow(cv2.imread(PREPROCESSED_IMG))
    fig3 = px.imshow(cv2.imread(OCR_OUTPUT_PRE))
    fig.show()
    fig2.show()
    fig3.show()
    # img_ori = cv2.imread(INPUT_IMAGE)
    # img_pre = cv2.imread(PREPROCESSED_IMG)
    # cv2.imshow("Original", cv2.resize(img_ori, (0, 0), fx=0.4, fy=0.4))
    # cv2.imshow("Preprocessed", cv2.resize(img_pre, (0, 0), fx=0.4, fy=0.4))
    # print("กดปุ่มใดก็ได้บนคีย์บอร์ดเพื่อปิดหน้าต่างภาพ...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
