path = r"d:\Document\Year4_Semester1\KhoaLuanTotNghiep\FoodDetector-2 - V3\depth_estimation.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

target = """        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth.float()"""

repl = """        with torch.no_grad():
            cudnn_prev = torch.backends.cudnn.enabled
            try:
                # Disable cuDNN convolution search to avoid RuntimeError on Turing cards
                torch.backends.cudnn.enabled = False
                outputs = self._model(**inputs)
            finally:
                torch.backends.cudnn.enabled = cudnn_prev
            predicted_depth = outputs.predicted_depth.float()"""

text_norm = text.replace("\r\n", "\n")
target_norm = target.replace("\r\n", "\n")
repl_norm = repl.replace("\r\n", "\n")

assert target_norm in text_norm, "Target not found in depth_estimation.py"
text_norm = text_norm.replace(target_norm, repl_norm, 1)

with open(path, "w", encoding="utf-8", newline="\r\n") as f:
    f.write(text_norm)
print("depth_estimation.py updated successfully!")
