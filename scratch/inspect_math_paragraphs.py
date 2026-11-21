import docx

doc = docx.Document(r'd:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\KLTN_Chuong1-3_bien_tap_ngon_ngu.docx')

with open('scratch/math_paragraphs_inspection.txt', 'w', encoding='utf-8') as out:
    for idx in [290, 291, 348]:
        p = doc.paragraphs[idx]
        out.write(f"=== Paragraph {idx} ===\n")
        for child in p._element:
            tag = child.tag.split('}')[-1]
            if tag == 'r':
                t = child.xpath('.//*[local-name()="t"]')
                text = t[0].text if t else ''
                out.write(f"  [Run]: {repr(text)}\n")
            elif tag == 'oMath':
                m_t = child.xpath('.//*[local-name()="t"]')
                m_text = ''.join([elem.text for elem in m_t if elem.text])
                out.write(f"  [Math]: {repr(m_text)}\n")
            else:
                out.write(f"  [{tag}]\n")
print("Inspection written to scratch/math_paragraphs_inspection.txt")
