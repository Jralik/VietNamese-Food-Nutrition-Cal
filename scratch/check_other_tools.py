import requests
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

for path in ['nhu-cau-dinh-duong', 'danh-gia-tinh-trang-dinh-duong']:
    url = f"https://viendinhduong.vn/vi/cong-cu-va-tien-ich/{path}"
    r = requests.get(url, timeout=10)
    print(f"=== {path} ===")
    scripts = re.findall(r'<script src="([^"]+)"', r.text)
    for s in scripts:
        if 'bundle' in s or 'plugin' in s:
            print("Script:", s)
