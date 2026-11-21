import docx

doc = docx.Document(r'd:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\KLTN_Chuong1-3_bien_tap_ngon_ngu.docx')

with open('scratch/ch3_all_paragraphs.txt', 'w', encoding='utf-8') as out:
    for i in range(287, 383):
        p = doc.paragraphs[i]
        out.write(f"=== P[{i}] (Style: {p.style.name}) ===\n{p.text}\n\n")

print(f"Dumped paragraphs 287 to 382. Total: {383 - 287}")
