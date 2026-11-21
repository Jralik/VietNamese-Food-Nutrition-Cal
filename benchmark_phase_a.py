"""
Phase A Verification Benchmark:
Tests volume_integration.estimate_nutrition_volume with Persistent Worker Daemon across 3 calls.
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

print("--- Calling estimate_nutrition_volume (Call 1) ---")
t0 = time.time()
res1 = volume_integration.estimate_nutrition_volume(img, dets, backend="sam2")
t1 = time.time() - t0

print("--- Calling estimate_nutrition_volume (Call 2) ---")
t0 = time.time()
res2 = volume_integration.estimate_nutrition_volume(img, dets, backend="sam2")
t2 = time.time() - t0

print("--- Calling estimate_nutrition_volume (Call 3) ---")
t0 = time.time()
res3 = volume_integration.estimate_nutrition_volume(img, dets, backend="sam2")
t3 = time.time() - t0

print("\n" + "="*60)
print("PHASE A PERSISTENT WORKER END-TO-END BENCHMARK")
print("="*60)
print(f"Call 1 (Daemon init + 1st inference): wall={t1:.3f}s | worker={res1.worker_elapsed_s}s | mode={res1.worker_mode} | fallback={res1.worker_fallback} | pid={res1.worker_pid}")
print(f"Call 2 (Warm daemon request):          wall={t2:.3f}s | worker={res2.worker_elapsed_s}s | mode={res2.worker_mode} | fallback={res2.worker_fallback} | pid={res2.worker_pid}")
print(f"Call 3 (Warm daemon request):          wall={t3:.3f}s | worker={res3.worker_elapsed_s}s | mode={res3.worker_mode} | fallback={res3.worker_fallback} | pid={res3.worker_pid}")
print(f"Consistency check: Vol={res1.estimations[0].volume_cm3:.1f} cm3, Mass={res1.estimations[0].mass_g:.1f}g, Calories={res1.estimations[0].nutrition.get('Calories', 0):.0f} kcal")
