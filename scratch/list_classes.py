import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from class_names import class_names
for i, c in enumerate(class_names):
    print(f"{i:02d}: {c.get('name')}")
