from __future__ import annotations
from typing import List, Dict
from datetime import date
import httpx
import pandas as pd
import json
from pathlib import Path
from api.config import settings

class BaseProvider:
    async def fetch(self, keyword: str, engine: str, as_of: date) -> List[Dict]:
        """Return list of dicts: {'rank': int, 'url': str, 'snippet': str, 'serp_features': {...}}"""
        raise NotImplementedError

class SerpAPIProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://serpapi.example"  # replace with real provider base URL if used

    async def fetch(self, keyword: str, engine: str, as_of: date):
        params = {"q": keyword, "engine": engine, "api_key": self.api_key}
        async with httpx.AsyncClient(timeout=30) as client:
            # Example generic call: adapt to the provider you use
            r = await client.get(self.base + "/search", params=params)
            r.raise_for_status()
            payload = r.json()
        # Map payload -> list of rank results (this mapping depends on provider)
        results = []
        # Example mapping (provider-specific keys will differ)
        for i, item in enumerate(payload.get("organic_results", []), start=1):
            results.append({
                "rank": item.get("position", i),
                "url": item.get("link") or item.get("url"),
                "snippet": item.get("snippet"),
                "serp_features": item.get("serp_features", {})
            })
        return results

class OfflineCSVProvider(BaseProvider):
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)

    async def fetch(self, keyword: str, engine: str, as_of: date):
        # file format: data/YYYY-MM-DD.csv with columns: keyword, engine, rank, url, snippet, serp_features (json)
        csv_file = self.data_dir / f"{as_of.isoformat()}.csv"
        if not csv_file.exists():
            return []
        df = pd.read_csv(csv_file)
        # filter by keyword and engine
        df = df[(df["keyword"].astype(str).str.lower() == keyword.lower()) & (df["engine"] == engine)]
        results = []
        for _, row in df.iterrows():
            serp_features = {}
            if "serp_features" in row and pd.notna(row["serp_features"]):
                try:
                    serp_features = json.loads(row["serp_features"])
                except Exception:
                    serp_features = row["serp_features"]
            results.append({
                "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
                "url": row.get("url"),
                "snippet": row.get("snippet"),
                "serp_features": serp_features
            })
        return results

def get_provider() -> BaseProvider:
    if settings.SERP_API_KEY:
        return SerpAPIProvider(settings.SERP_API_KEY)
    return OfflineCSVProvider()
