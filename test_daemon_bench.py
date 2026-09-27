"""
Benchmark Persistent Daemon Worker for SAM2 Box-Prompted.
Measures Cold startup vs Request 1 vs Request 2 vs Request 3.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
WORKER_SCRIPT = str(PROJECT_ROOT / "volume_worker.py")
IMG_PATH = r"C:\Users\huynh\OneDrive\Pictures\test-image\No_ArUcO\Pho.jpg"

job = {
    "image_path": IMG_PATH,
    "detections": [
        {"bbox": [67, 159, 833, 673], "class_name": "Pho (Vietnamese noodle soup)", "confidence": 0.96}
    ],
    "backend": "sam2"
}

print("=== STARTING DAEMON PROCESS ===")
t_start = time.time()
proc = subprocess.Popen(
    [PYTHON_EXE, WORKER_SCRIPT, "--daemon"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=sys.stderr,
    text=True,
    bufsize=1,
    cwd=str(PROJECT_ROOT)
)

ready_line = proc.stdout.readline()
t_ready = time.time() - t_start
print(f"Daemon pre-warmed and ready in {t_ready:.2f}s: {ready_line.strip()}")

# Request 1
print("\n--- Sending Request 1 ---")
t0 = time.time()
proc.stdin.write(json.dumps(job) + "\n")
proc.stdin.flush()
line1 = proc.stdout.readline()
t_req1 = time.time() - t0
res1 = json.loads(line1)

# Request 2
print("--- Sending Request 2 ---")
t0 = time.time()
proc.stdin.write(json.dumps(job) + "\n")
proc.stdin.flush()
line2 = proc.stdout.readline()
t_req2 = time.time() - t0
res2 = json.loads(line2)

# Request 3
print("--- Sending Request 3 ---")
t0 = time.time()
proc.stdin.write(json.dumps(job) + "\n")
proc.stdin.flush()
line3 = proc.stdout.readline()
t_req3 = time.time() - t0
res3 = json.loads(line3)

proc.stdin.close()
proc.wait(timeout=5)

print("\n" + "="*55)
print("PERSISTENT WORKER EMPIRICAL RESULTS (SAM2 BOX)")
print("="*55)
print(f"Daemon Initial Startup & Warm Preload: {t_ready:.2f}s")
print(f"Request 1: wall={t_req1:.3f}s | worker={res1.get('worker_elapsed_s')}s | mode={res1.get('worker_mode')} | vol={res1['estimations'][0]['volume_cm3']:.1f} cm3")
print(f"Request 2: wall={t_req2:.3f}s | worker={res2.get('worker_elapsed_s')}s | mode={res2.get('worker_mode')} | vol={res2['estimations'][0]['volume_cm3']:.1f} cm3")
print(f"Request 3: wall={t_req3:.3f}s | worker={res3.get('worker_elapsed_s')}s | mode={res3.get('worker_mode')} | vol={res3['estimations'][0]['volume_cm3']:.1f} cm3")
print(f"Worker PID: {res1.get('worker_pid')}")
