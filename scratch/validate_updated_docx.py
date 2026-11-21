import docx

doc = docx.Document(r'd:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\KLTN_Chuong1-3_bien_tap_ngon_ngu.docx')

with open('scratch/validation_report.txt', 'w', encoding='utf-8') as out:
    out.write(f"Total Paragraphs: {len(doc.paragraphs)}\n")
    out.write(f"Total Tables: {len(doc.tables)}\n\n")

    # Check math formulas
    out.write("=== MATH FORMULAS CHECK (Chapter 3) ===\n")
    math_count = 0
    for i in range(287, 382):
        p = doc.paragraphs[i]
        m_elems = p._element.xpath('.//*[local-name()="oMath"]')
        if m_elems:
            math_count += len(m_elems)
            m_texts = [''.join([t.text for t in elem.xpath('.//*[local-name()="t"]') if t.text]) for elem in m_elems]
            out.write(f"P[{i}]: {len(m_elems)} math element(s): {m_texts}\n")
    out.write(f"Total math elements in Chapter 3: {math_count}\n\n")

    # Check headings
    out.write("=== HEADINGS CHECK (Chapter 3) ===\n")
    for i in range(287, 382):
        p = doc.paragraphs[i]
        txt = p.text.strip()
        if txt.startswith("3.") or txt.startswith("CHƯƠNG 3"):
            out.write(f"P[{i}] ({p.style.name}): {txt}\n")

    # Check figures and tables captions
    out.write("\n=== CAPTIONS CHECK (Chapter 3) ===\n")
    for i in range(287, 382):
        p = doc.paragraphs[i]
        txt = p.text.strip()
        if txt.startswith("Hình 3.") or txt.startswith("Bảng 3."):
            out.write(f"P[{i}]: {txt}\n")

    # Check key phrases
    out.write("\n=== KEY PHRASES CHECK ===\n")
    ch3_text = "\n".join([doc.paragraphs[i].text for i in range(287, 382)])
    out.write(f"Contains 'marker ArUco': {'marker ArUco' in ch3_text}\n")
    out.write(f"Contains 'khẩu phần ăn mẫu chuẩn': {'khẩu phần ăn mẫu chuẩn' in ch3_text}\n")
    out.write(f"Contains 'reference_serving_fallback': {'reference_serving_fallback' in ch3_text}\n")
    out.write(f"Contains 'Lý do loại bỏ phương pháp phỏng đoán theo đĩa': {'Lý do loại bỏ phương pháp phỏng đoán theo đĩa' in ch3_text}\n")
    out.write(f"Contains old 'dish_geometry/plate_heuristic': {'dish_geometry/plate_heuristic' in ch3_text}\n")
    out.write(f"Contains old 'plate_heuristic': {'plate_heuristic' in ch3_text}\n")

print("Validation script completed")
