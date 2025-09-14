import asyncio
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import select, func
from statsmodels.tsa.seasonal import STL
from api.db import AsyncSessionLocal
from api.models import Keyword, Rank, Anomaly, SerpProvider, Engine
from api.providers import get_provider
from api.config import settings
from slack_sdk.webhook import WebhookClient

async def fetch_daily(as_of: date | None = None):
    as_of = as_of or date.today()
    provider = get_provider()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Keyword).options())
        keywords = result.scalars().all()
        # Optionally create provider entry in DB
        # provider_db = SerpProvider(name=provider.__class__.__name__)
        # session.add(provider_db); await session.commit()

        for kw in keywords:
            try:
                rows = await provider.fetch(kw.text, kw.engine.name, as_of)
            except Exception as e:
                # log and continue
                print("provider error", e)
                continue

            for r in rows:
                rank_obj = Rank(
                    keyword_id=kw.id,
                    date=as_of,
                    rank=r.get("rank"),
                    url=r.get("url"),
                    snippet=r.get("snippet"),
                    serp_features=r.get("serp_features"),
                )
                session.add(rank_obj)
        await session.commit()

async def compute_volatility(window_days=(7, 30), lookback_days=180):
    """Compute volatility (std of rank differences) and store on keyword.vol_7 / vol_30"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Keyword))
        keywords = result.scalars().all()
        for kw in keywords:
            # fetch last N days of ranks
            stmt = select(Rank.date, Rank.rank).where(Rank.keyword_id == kw.id).order_by(Rank.date)
            res = await session.execute(stmt)
            rows = res.all()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["date", "rank"])
            df = df.dropna(subset=["rank"]).set_index("date").sort_index()
            if df.empty or len(df) < 2:
                continue
            # compute day-to-day differences
            diff = df["rank"].diff().dropna()
            # use rolling std over window_days
            vols = {}
            for w in window_days:
                if len(diff) >= w:
                    vols[f"vol_{w}"] = float(diff.rolling(window=w).std().iloc[-1])
                else:
                    vols[f"vol_{w}"] = float(diff.std())
            # update keyword
            kw.vol_7 = vols.get("vol_7")
            kw.vol_30 = vols.get("vol_30")
            session.add(kw)
        await session.commit()

def _mad_zscore(arr: np.ndarray):
    """Robust z-score using MAD"""
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    # scale constant for normal dist
    denom = 1.4826 * mad if mad != 0 else np.std(arr) or 1.0
    return (arr - med) / denom

async def detect_anomalies(min_length=14, z_threshold=3.5, as_of: date | None = None):
    as_of = as_of or date.today()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Keyword))
        keywords = result.scalars().all()
        webhook = WebhookClient(settings.SLACK_WEBHOOK_URL) if settings.SLACK_WEBHOOK_URL else None

        for kw in keywords:
            stmt = select(Rank.date, Rank.rank).where(Rank.keyword_id == kw.id).order_by(Rank.date)
            res = await session.execute(stmt)
            rows = res.all()
            if len(rows) < min_length:
                continue
            df = pd.DataFrame(rows, columns=["date", "rank"]).dropna()
            df = df.set_index("date").asfreq("D").interpolate()  # daily series
            series = df["rank"].astype(float)

            # STL decomposition - weekly seasonality (period=7)
            try:
                stl = STL(series, period=7, robust=True)
                res = stl.fit()
                resid = res.resid.values
            except Exception:
                # fallback: resid = series - rolling median
                resid = series - series.rolling(7, min_periods=1).median()
                resid = resid.values

            z = _mad_zscore(resid)
            anomalous_idx = np.where(np.abs(z) > z_threshold)[0]
            if anomalous_idx.size == 0:
                continue

            # insert anomalies and optionally alert
            for idx in anomalous_idx:
                d = df.index[idx].date()
                score = float(z[idx])
                payload = {"recent_rank": float(df.iloc[idx]["rank"])}
                anomaly = Anomaly(keyword_id=kw.id, date=d, score=score, method="stl_mad_z", payload=payload)
                session.add(anomaly)
            await session.commit()

            if webhook:
                text = f"Anomalies detected for keyword `{kw.text}`: {len(anomalous_idx)} on or before {as_of.isoformat()}"
                webhook.send(text=text)
