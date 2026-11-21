"""
scripts/sync_tdgd_data.py
=========================
ETL script to sync the Vietnamese Food Composition Table from the
"Thực đơn gia đình" portal (Viện Nghiên Cứu Dinh Dưỡng TP.HCM — Nuti Expert):

1. Food composition lookup (Tra cứu thành phần dinh dưỡng thực phẩm):
   https://app.thucdongiadinh.vn/api/services/app/TraCuuDinhDuong/TraCuuDinhDuongThucPham
   - Basis: per_100g (khoiLuong = 100.0)
   - Source table: Bảng thành phần thực phẩm Việt Nam (mã BTP2007/BTP2000)
   - Output: data/tdgd_food_composition.json

This dataset is the SECOND authoritative VN source used to cross-check the
NIN (viendinhduong.vn) canonical values in data/nin_nutrition_canonical.json.

Usage: python scripts/sync_tdgd_data.py
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
    "Content-Type": "application/json",
    "Origin": "https://thucdongiadinh.vn",
    "Referer": "https://thucdongiadinh.vn/",
}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FOOD_API = "https://app.thucdongiadinh.vn/api/services/app/TraCuuDinhDuong/TraCuuDinhDuongThucPham"
DISH_API = "https://app.thucdongiadinh.vn/api/services/app/MonAn/GetAllServerPaging"


def fetch_all(url: str, page_size: int = 200, max_retries: int = 3) -> list:
    """Fetch every record from an ABP-style skipCount/maxResultCount endpoint."""
    all_items = []
    skip = 0
    total = None

    while True:
        payload = {"skipCount": skip, "maxResultCount": page_size, "filter": ""}
        items = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.post(url, headers=HEADERS, json=payload, timeout=30)
                if resp.status_code == 200:
                    result = resp.json().get("result", {})
                    items = result.get("items", [])
                    if total is None:
                        total = result.get("totalCount", 0)
                        logger.info(f"Endpoint reported total: {total} records.")
                    break
                logger.warning(f"skipCount={skip} returned status {resp.status_code} (attempt {attempt}/{max_retries})")
            except Exception as e:
                logger.warning(f"Error fetching skipCount={skip} on attempt {attempt}: {e}")
            time.sleep(1.0)

        if items is None:
            logger.error(f"Failed to fetch skipCount={skip} after {max_retries} attempts.")
            break
        if not items:
            break

        all_items.extend(items)
        skip += len(items)
        logger.info(f"Fetched {len(all_items)}/{total} records")

        if total and skip >= total:
            break
        time.sleep(0.3)

    return all_items


def sync_food_composition():
    """Sync the 1,800+ food composition table (per-100g basis)."""
    logger.info("=== 1/2: Syncing TDGD Food Composition Table ===")
    items = fetch_all(FOOD_API)

    output_path = os.path.join(DATA_DIR, "tdgd_food_composition.json")
    payload = {
        "metadata": {
            "source": "Bảng thành phần thực phẩm Việt Nam — Cổng Thực đơn gia đình (thucdongiadinh.vn), Viện Nghiên Cứu Dinh Dưỡng TP.HCM",
            "api_endpoint": FOOD_API,
            "basis_original": "per_100g",
            "food_table_codes": ["BTP2007", "BTP2000"],
            "description": "Bảng thành phần dinh dưỡng thực phẩm Việt Nam (tra cứu công khai, cơ sở 100g phần ăn được)",
            "total_records": len(items),
            "synced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": items,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully saved {len(items)} foods to {output_path}")
    return len(items)


def sync_dishes():
    """Sync the dish database (mostly branded/commercial products)."""
    logger.info("=== 2/2: Syncing TDGD Dish Database ===")
    items = fetch_all(DISH_API, page_size=100)

    output_path = os.path.join(DATA_DIR, "tdgd_dishes.json")
    payload = {
        "metadata": {
            "source": "Cơ sở dữ liệu món ăn — Cổng Thực đơn gia đình (thucdongiadinh.vn), Viện Nghiên Cứu Dinh Dưỡng TP.HCM",
            "api_endpoint": DISH_API,
            "basis_original": "per_serving",
            "description": "Món ăn kèm hàm lượng dinh dưỡng theo khẩu phần (phần lớn là sản phẩm chế biến sẵn thương mại)",
            "total_records": len(items),
            "synced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": items,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully saved {len(items)} dishes to {output_path}")
    return len(items)


if __name__ == "__main__":
    t0 = time.time()
    n_foods = sync_food_composition()
    n_dishes = sync_dishes()
    logger.info(f"All datasets synced successfully in {time.time() - t0:.2f}s! ({n_foods} foods, {n_dishes} dishes)")
