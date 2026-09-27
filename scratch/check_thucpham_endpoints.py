import requests
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "https://viendinhduong.vn/plugins/manager/vdd/assets/dist/js/vdd/nutritionalFood/pageIndex/bundle.js"
r = requests.get(url, timeout=15)
text = r.text

matches = re.findall(r'["\'](/[^"\'\s<>]+)["\']', text)
interesting = [m for m in set(matches) if any(k in m for k in ['api', 'food', 'nutri', 'tool'])]
print("Interesting:", interesting)

# Search for api
matches_api = re.findall(r'(https?://[^"\'\s<>]+)', text)
print("HTTP matches:", set(matches_api))

# Search for get or post
calls = re.findall(r'([A-Za-z0-9_]+\.(?:get|post)\([^\)]+\))', text)
print("Calls count:", len(calls))
if calls:
    print("Sample calls:", calls[:5])
