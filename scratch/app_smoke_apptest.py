"""Smoke test through the real Streamlit app (main.py) via AppTest.

Simulates: app render -> file upload -> "Dự đoán" click -> detection cards.
Exercises the new predict_with_dual_preresize path inside the actual app
code, including detect_image_result rendering and the volume pipeline call.
"""

import os
import sys

sys.path.insert(0, os.getcwd())  # main.py imports ui_components etc. from project root

from streamlit.testing.v1 import AppTest

IMAGES = [
    r"C:\Users\huynh\OneDrive\Pictures\test-image\SupCua\SupCua (1).jpg",
    r"C:\Users\huynh\OneDrive\Pictures\test-image\BanhMi\BanhMi_Trung (4).jpg",
]


def pick_uploader(at):
    for fu in at.file_uploader:
        types = set(fu.allowed_type or [])
        if types and any(t.lower().startswith(("jpg", "png", "jpeg")) for t in types):
            return fu
    return at.file_uploader[0] if at.file_uploader else None


for path in IMAGES:
    print(f"\n================ APPTEST SMOKE: {path.split(chr(92))[-1]} ================")
    at = AppTest.from_file("main.py", default_timeout=1200)
    at.run(timeout=300)
    if at.exception:
        print("EXCEPTION ON RENDER:", [e.value for e in at.exception])
        sys.exit(1)
    print("render OK")

    # switch input source to file upload (radio defaults to demo mode)
    radios = [r for r in at.radio if "Nguồn ảnh" in (r.label or "")]
    if not radios:
        print("NO SOURCE RADIO:", [r.label for r in at.radio])
        sys.exit(1)
    radios[0].set_value("📁 Tải ảnh lên")
    at.run(timeout=300)
    if at.exception:
        print("EXCEPTION AFTER MODE SWITCH:", [e.value for e in at.exception])
        sys.exit(1)

    fu = pick_uploader(at)
    if fu is None:
        print("NO FILE UPLOADER FOUND; uploaders:", [u.label for u in at.file_uploader])
        sys.exit(1)
    print("uploader label:", fu.label)
    with open(path, "rb") as f:
        fu.upload(os.path.basename(path), f.read(),
                  mime_type="image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png")
    at.run(timeout=300)
    if at.exception:
        print("EXCEPTION AFTER UPLOAD:", [e.value for e in at.exception])
        sys.exit(1)
    print("upload OK")

    btns = [b for b in at.button if "Dự đoán" in (b.label or "")]
    if not btns:
        print("NO PREDICT BUTTON; buttons:", [b.label for b in at.button][:10])
        sys.exit(1)
    btns[0].click()
    at.run(timeout=1200)
    if at.exception:
        print("EXCEPTION AFTER PREDICT:", [e.value for e in at.exception])
        sys.exit(1)

    texts = " | ".join(
        (m.value or "") for m in list(at.markdown) + list(at.subheader) + list(at.header)
    )
    hits = [k for k in ("Sup cua", "Banh mi", "Không phát hiện", "khẩu phần") if k in texts]
    print("PREDICT OK — key texts found:", hits)
    err_msgs = [e.value for e in at.error]
    if err_msgs:
        print("st.error messages:", err_msgs[:3])
print("\nSMOKE TEST DONE")
