"""Konfigurasi pytest: tambahkan root repo ke sys.path agar `src` dan `scripts` bisa diimpor."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
