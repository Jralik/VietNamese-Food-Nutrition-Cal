import re
from pathlib import Path

script_path = Path(r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\eval_segmentation_benchmark.py")
with open(script_path, "r", encoding="utf-8") as f:
    code = f.read()

# Let's inspect the function definition
func_start = code.find("def generate_markdown_report(")
assert func_start != -1, "generate_markdown_report not found"
func_end = code.find("def main():", func_start)
assert func_end != -1, "def main() not found"

print(f"generate_markdown_report spans from {func_start} to {func_end}")

