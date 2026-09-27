"""
Phase B Verification Benchmark:
Tests End-to-End FoodSAM Volume & Nutrition Estimation across 3 consecutive calls
with Persistent Daemon Worker enabled.
"""
import sys
import time
from PIL import Image
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import volume_integration

IMG_PATH = r"C:\Users\huynh\OneDrive\Pictures\test-image\No_ArUcO\Pho.jpg"
img = Image.open(IMG_PATH).convert("RGB")
dets = [{"bbox": [67, 159, 833, 673], "class_name": "Pho (Vietnamese noodle soup)", "confidence": 0.96}]

print("================================================================")
print("PHASE B: FOODSAM PERSISTENT DAEMON END-TO-END BENCHMARK")
print("================================================================")

print("\n--- Calling estimate_nutrition_volume (Call 1 - Cold init + 1st inference) ---")
t0 = time.time()
res1 = volume_integration.estimate_nutrition_volume(img, dets, backend="foodsam")
t1 = time.time() - t0
print(f"Call 1 done: wall={t1:.3f}s | worker_elapsed={res1.worker_elapsed_s if res1 else 'N/A'}s | fallback={res1.worker_fallback if res1 else 'N/A'}")

print("\n--- Calling estimate_nutrition_volume (Call 2 - Warm daemon) ---")
t0 = time.time()
res2 = volume_integration.estimate_nutrition_volume(img, dets, backend="foodsam")
t2 = time.time() - t0
print(f"Call 2 done: wall={t2:.3f}s | worker_elapsed={res2.worker_elapsed_s if res2 else 'N/A'}s | fallback={res2.worker_fallback if res2 else 'N/A'}")

print("\n--- Calling estimate_nutrition_volume (Call 3 - Warm daemon) ---")
t0 = time.time()
res3 = volume_integration.estimate_nutrition_volume(img, dets, backend="foodsam")
t3 = time.time() - t0
print(f"Call 3 done: wall={t3:.3f}s | worker_elapsed={res3.worker_elapsed_s if res3 else 'N/A'}s | fallback={res3.worker_fallback if res3 else 'N/A'}")

print("\n" + "="*65)
print("FINAL BENCHMARK SUMMARY TABLE (FOODSAM ON GTX 1650 4GB)")
print("="*65)
for idx, (label, res, wall) in enumerate([("Call 1 (Cold+Init)", res1, t1),
                                          ("Call 2 (Warm)", res2, t2),
                                          ("Call 3 (Warm)", res3, t3)], start=1):
    if res is not None:
        n_dish = sum(1 for e in res.estimations if not e.is_ingredient)
        n_extra = sum(1 for e in res.estimations if e.is_ingredient)
        tot_cal = res.total_nutrition.get('Calories', 0)
        tot_mass = sum(e.mass_g for e in res.estimations)
        print(f"{label:<22} | Wall: {wall:>6.2f}s | Worker: {res.worker_elapsed_s:>6.2f}s | Dishes: {n_dish} | Extras: {n_extra} | Mass: {tot_mass:>6.1f}g | Cal: {tot_cal:>5.0f} kcal")
    else:
        print(f"{label:<22} | FAILED")
print("="*65)
