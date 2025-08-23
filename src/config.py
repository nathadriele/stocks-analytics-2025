"""
Configurações globais do projeto Stock Market Analytics Zoomcamp 2025.

Carrega variáveis do .env e define caminhos principais.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env", override=True)

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANALYTICS_DIR = DATA_DIR / "analytics"
SIGNALS_DIR = DATA_DIR / "signals"
BACKTESTS_DIR = DATA_DIR / "backtests"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
IMG_DIR = REPORTS_DIR / "img"
STORAGE_DIR = BASE_DIR / "storage"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

for d in [
    RAW_DIR,
    PROCESSED_DIR,
    ANALYTICS_DIR,
    SIGNALS_DIR,
    BACKTESTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    IMG_DIR,
    STORAGE_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

DATA_START = os.getenv("DATA_START", "2015-01-01")
TICKERS = [t.strip() for t in os.getenv("TICKERS", "AAPL,MSFT,SPY").split(",") if t.strip()]
DB_PATH = os.getenv("DB_PATH", str(STORAGE_DIR / "app.db"))

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")

SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")
