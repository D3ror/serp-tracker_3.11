# api/main.py
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.models import Base
from api.jobs import fetch_daily, compute_volatility, detect_anomalies
from api.config import settings  # loads DATABASE_URL, etc.

# --- Database setup ---
# Use sync driver for table creation
SYNC_DB_URL = settings.DATABASE_URL.replace("+asyncpg", "")
engine = create_engine(SYNC_DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Create tables if they don't exist ---
Base.metadata.create_all(bind=engine)

# --- FastAPI app ---
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/fetch-daily")
def run_fetch():
    fetch_daily()
    return {"status": "ok"}

@app.post("/compute")
def run_compute():
    compute_volatility()
    detect_anomalies()
    return {"status": "ok"}
