import os
import sys
from pathlib import Path

os.environ["MOMO_SKIP_MODEL_BOOTSTRAP"] = "1"
os.environ["MOMO_SKIP_TTS_BENCHMARK"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
