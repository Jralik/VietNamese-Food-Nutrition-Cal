import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Referer": "https://viendinhduong.vn"
}

# Check dishes pagination
r_dishes = requests.get("https://viendinhduong.vn/api/fe/tool/getPageFoodData?page=1&pageSize=100", headers=headers, timeout=10)
d_data = r_dishes.json()
print("Dishes total:", d_data.get("total"), "| Last page at pageSize=100:", d_data.get("last_page"), "| Items in page 1:", len(d_data.get("data", [])))

# Check raw food pagination
r_raw = requests.get("https://viendinhduong.vn/api/fe/foodNatunal/getPageFoodData?page=1&pageSize=100", headers=headers, timeout=10)
raw_data = r_raw.json()
print("Raw food total:", raw_data.get("total"), "| Last page at pageSize=100:", raw_data.get("last_page"), "| Items in page 1:", len(raw_data.get("data", [])))
