"""
scripts/sync_nin_data.py
========================
ETL script to sync raw nutrition datasets from the National Institute of Nutrition (NIN):
1. Dishes API: https://viendinhduong.vn/api/fe/tool/getPageFoodData
   - Basis: per_serving (1 portion)
   - Output: data/nin_dishes.json

2. Raw Foods (Ingredients) API: https://viendinhduong.vn/api/fe/foodNatunal/getPageFoodData
   - Basis: per_100g (100g edible portion)
   - Output: data/nin_ingredients.json

Acceptance Criteria verified:
- P1-A: nin_dishes.json preserves raw data and marks basis_original = "per_serving"
- P1-B: nin_ingredients.json preserves raw data and marks basis_original = "per_100g"
"""

import os
import sys
import time
import json
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://viendinhduong.vn"
}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_paginated(url: str, page_size: int = 100, max_retries: int = 3, delay: float = 0.3) -> list:
    """Fetch all pages from a paginated NIN endpoint."""
    all_items = []
    page = 1
    total = None
    
    while True:
        target_url = f"{url}?page={page}&pageSize={page_size}"
        success = False
        
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(target_url, headers=HEADERS, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    success = True
                    break
                else:
                    logger.warning(f"Page {page} returned status {resp.status_code} (attempt {attempt}/{max_retries})")
            except Exception as e:
                logger.warning(f"Error fetching page {page} on attempt {attempt}: {e}")
            time.sleep(1.0)
            
        if not success:
            logger.error(f"Failed to fetch page {page} after {max_retries} attempts.")
            break
            
        items = data.get("data", [])
        if not items:
            break
            
        all_items.extend(items)
        if total is None:
            total = data.get("total", len(items))
            logger.info(f"Endpoint reported total: {total} records.")
            
        logger.info(f"Fetched page {page}: {len(items)} items (accumulated: {len(all_items)}/{total})")
        
        last_page = data.get("last_page")
        if last_page is not None and page >= last_page:
            break
        if len(all_items) >= total:
            break
            
        page += 1
        time.sleep(delay)
        
    return all_items


def sync_dishes():
    """Sync 1,250 cooked Vietnamese dishes."""
    logger.info("=== 1/2: Syncing NIN Dishes Database ===")
    api_url = "https://viendinhduong.vn/api/fe/tool/getPageFoodData"
    items = fetch_paginated(api_url, page_size=100)
    
    output_path = os.path.join(DATA_DIR, "nin_dishes.json")
    payload = {
        "metadata": {
            "source": "Viện Dinh dưỡng Quốc gia (viendinhduong.vn)",
            "api_endpoint": api_url,
            "basis_original": "per_serving",
            "description": "Cơ sở dữ liệu giá trị dinh dưỡng của các món ăn chế biến sẵn tại Việt Nam",
            "total_records": len(items),
            "synced_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "data": items
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully saved {len(items)} dishes to {output_path}")
    return len(items)


def sync_ingredients():
    """Sync raw food composition tables (NIN FoodNatural)."""
    logger.info("=== 2/2: Syncing NIN Raw Ingredients Database ===")
    api_url = "https://viendinhduong.vn/api/fe/foodNatunal/getPageFoodData"
    items = fetch_paginated(api_url, page_size=100)
    
    output_path = os.path.join(DATA_DIR, "nin_ingredients.json")
    payload = {
        "metadata": {
            "source": "Bảng thành phần thực phẩm Việt Nam - Viện Dinh dưỡng Quốc gia",
            "api_endpoint": api_url,
            "basis_original": "per_100g",
            "description": "Bảng thành phần thực phẩm nguyên liệu tự nhiên (thực vật, động vật)",
            "total_records": len(items),
            "synced_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "data": items
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully saved {len(items)} raw ingredients to {output_path}")
    return len(items)


if __name__ == "__main__":
    t0 = time.time()
    n_dishes = sync_dishes()
    n_ingredients = sync_ingredients()
    logger.info(f"All datasets synced successfully in {time.time() - t0:.2f}s! ({n_dishes} dishes, {n_ingredients} ingredients)")
