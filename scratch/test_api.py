import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Referer": "https://viendinhduong.vn/vi/cong-cu-va-tien-ich/gia-tri-dinh-duong-mon-an"
}

url = "https://viendinhduong.vn/api/fe/tool/getPageFoodData?page=1&pageSize=2"
r = requests.get(url, headers=headers, timeout=10)
res = r.json()
print("Total records:", res.get("total"))
print("First item keys:", list(res["data"][0].keys()))
print("First item sample:")
print(json.dumps(res["data"][0], indent=2, ensure_ascii=False))
