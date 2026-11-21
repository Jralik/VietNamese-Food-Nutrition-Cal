import docx
import sys

doc = docx.Document(r'd:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\KLTN_Chuong1-3_bien_tap_ngon_ngu.docx')

keywords = ['plate', 'heuristic', 'aruco', 'khẩu phần', 'thay thế', 'đế đựng', 'tham chiếu', 'fallback']
in_ch3 = False
with open('scratch/ch3_keyword_hits.txt', 'w', encoding='utf-8') as out:
    for i, p in enumerate(doc.paragraphs):
        if 'CHƯƠNG 3' in p.text.upper():
            in_ch3 = True
        if 'TÀI LIỆU THAM KHẢO' in p.text.upper() or 'CHƯƠNG 4' in p.text.upper():
            in_ch3 = False
        if in_ch3:
            for kw in keywords:
                if kw in p.text.lower():
                    out.write(f'P[{i}] (found "{kw}"):\n{p.text}\n\n')
                    break
print("Done searching")
