from fastapi import FastAPI, Depends, Header, HTTPException, status
import asyncio
from api.config import settings
from api.jobs import fetch_daily, compute_volatility, detect_anomalies
from api.db import get_db

app = FastAPI(title="SERP Tracker")

def _check_token(token: str | None):
    if settings.API_AUTH_TOKEN and token != settings.API_AUTH_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bad token")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/fetch-daily")
async def run_fetch(x_api_token: str | None = Header(None)):
    _check_token(x_api_token)
    await fetch_daily()
    return {"status":"ok"}

@app.post("/compute")
async def run_compute(x_api_token: str | None = Header(None)):
    _check_token(x_api_token)
    await compute_volatility()
    await detect_anomalies()
    return {"status":"ok"}