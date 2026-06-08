from pathlib import Path
import os
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STANDARDIZED_DIR = DATA_DIR / "standardized"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks_with_embeddings.json"

if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "8"))
