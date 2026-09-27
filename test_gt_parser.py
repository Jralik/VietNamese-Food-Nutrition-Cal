"""Unit tests for the GT parser semantics (approved protocol STEP 4).

Locks:
  - "(tong)": grams_total = TOTAL of K boxes — must NOT be divided by K.
  - "?": count-only target — grams_total None, excluded from mass error.
  - Legacy formats parse with count=1 semantics "total".
"""

import os

import av  # noqa: F401  (venv DLL quirk guard)

from eval_full_pipeline import parse_gt, VAL_FILE

PASS = True


def check(name, cond):
    global PASS
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        PASS = False


def main():
    entries = dict(parse_gt(VAL_FILE))
    print(f"parsed {len(entries)} images")
    check("54 images parsed", len(entries) == 54)
    check("no empty GT", all(v for v in entries.values()))

    def gt(img, short):
        from class_names import class_names
        full = next(p for p in entries if p.replace("\\", "/").endswith(img))
        for cid, v in entries[full].items():
            if class_names[cid]["name"].split(" (")[0] == short:
                return v
        return None

    print("\n— (tong) semantics —")
    v = gt("BanhMi_Trung (1).jpg", "Banh mi")
    check("count == 2", v is not None and v["count"] == 2)
    check("grams_total == 99.3 (NOT divided by K)",
          v is not None and v["grams_total"] == 99.3)
    check("mass_semantics == 'total'", v is not None and v["mass_semantics"] == "total")
    v = gt("BanhMi_Trung (3).jpg", "Banh mi")
    check("BM(3): grams_total == 99.3", v is not None and v["grams_total"] == 99.3)

    print("\n— ? (count-only) semantics —")
    v = gt("com-suon1 (3).jpg", "Com tam")
    check("count == 2 (measured + unknown)", v is not None and v["count"] == 2)
    check("grams_total == 390.5 (measured only)",
          v is not None and v["grams_total"] == 390.5)
    check("mass_semantics == 'partial'", v is not None and v["mass_semantics"] == "partial")
    v = gt("com-suon1 (3).jpg", "Canh")
    check("Canh 1/150 unchanged", v is not None and v["count"] == 1
          and v["grams_total"] == 150.0)

    print("\n— legacy formats —")
    v = gt("banh-mi1.jpg", "Banh mi")
    check("banh-mi1: 1 / 242.6 total", v is not None and v["count"] == 1
          and v["grams_total"] == 242.6 and v["mass_semantics"] == "total")
    v = gt("SupCua1 (4).jpg", "Sup cua")
    check("SupCua1 (4): 1 / 500.6 total (numerical equivalence: "
          "old grams_each=G, count=1 -> new grams_total=G)",
          v is not None and v["count"] == 1 and v["grams_total"] == 500.6)
    v = gt("Pho1 (1).jpg", "Pho")
    check("Pho1 (1): 1 / 819.5", v is not None and v["count"] == 1
          and v["grams_total"] == 819.5)

    total = sum(v["count"] for img in entries.values() for v in img.values())
    print(f"\ntotal GT targets: {total}")
    print(f"\nOVERALL: {'ALL PASS' if PASS else 'FAIL'}")
    return PASS


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
