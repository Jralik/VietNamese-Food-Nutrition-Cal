import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "https://viendinhduong.vn/plugins/manager/vdd/assets/dist/js/vdd/tool/pageIndex/bundle.js?v=1.3.7"
r = requests.get(url, timeout=10)
text = r.text

pos = 0
for _ in range(5):
    pos = text.find('uC(', pos)
    if pos == -1:
        break
    print(f"=== uC call at {pos} ===")
    print(text[pos-100:pos+300])
    pos += 3
