import docx, sys
sys.stdout.reconfigure(encoding='utf-8')

doc = docx.Document(r'd:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\KLTN_Chuong1-3_bien_tap_ngon_ngu.docx')

print(f"Total paragraphs in doc: {len(doc.paragraphs)}")
# Chapter 3 spans from index 287 to 381
for i in range(287, 382):
    p = doc.paragraphs[i]
    style = p.style.name
    math_count = len(p._element.xpath('.//*[local-name()="oMath"]'))
    drawing_count = len(p._element.xpath('.//*[local-name()="drawing"]'))
    txt = p.text.strip()
    if txt.startswith("3.") or txt.startswith("CHƯƠNG") or txt.startswith("Hình") or txt.startswith("Bảng"):
        print(f"P[{i}] ({style}) [M:{math_count}, D:{drawing_count}]: {txt[:70]}")
