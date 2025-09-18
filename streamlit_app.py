import os
import re
import requests
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from api.config import settings
from api.models import reset_db   # ✅ reuse schema reset
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.seasonal import STL
import numpy as np

# -------------------
# Database Setup
# -------------------
def to_sync_url(async_url: str):
    return re.sub(r"\+asyncpg", "", async_url)

SYNC_DB_URL = to_sync_url(settings.DATABASE_URL)
engine = create_engine(SYNC_DB_URL, pool_pre_ping=True)

# ✅ Ensure schema exists
inspector = inspect(engine)
required_tables = {"keyword", "engine", "rank", "serp_feature", "anomaly"}
if not required_tables.issubset(set(inspector.get_table_names())):
    st.warning("⚠️ Database schema incomplete. Resetting...")
    reset_db()

SERP_API_KEY = settings.SERP_API_KEY

# -------------------
# Database Functions
# -------------------
@st.cache_data(ttl=600)
def load_keywords():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT id, text FROM keyword ORDER BY text", conn)
    return df

@st.cache_data
def fetch_rank_history(keyword_id: int):
    qry = text("""
       SELECT fetched_at AS date, position, domain
       FROM rank 
       WHERE keyword_id = :kid 
       ORDER BY fetched_at
    """)
    with engine.connect() as conn:
        df = pd.read_sql(qry, conn, params={"kid": keyword_id})
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    return df

def insert_keyword(kw: str):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO keyword (text) VALUES (:kw) ON CONFLICT DO NOTHING"),
            {"kw": kw},
        )

def delete_keyword(kw_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM keyword WHERE id=:id"), {"id": kw_id})

# -------------------
# SerpAPI Fetch
# -------------------
def fetch_serp_results(keyword: str, keyword_id: int):
    url = "https://serpapi.com/search"
    params = {
        "q": keyword,
        "engine": "google",
        "num": 100,
        "api_key": SERP_API_KEY,
    }
    r = requests.get(url, params=params)
    data = r.json()
    if "organic_results" not in data:
        return []

    rows = []
    for res in data["organic_results"]:
        pos = res.get("position")
        link = res.get("link", "")
        domain = re.sub(r"^https?://(www\.)?", "", link).split("/")[0] if link else "unknown"
        if pos:
            rows.append({
                "keyword_id": keyword_id,
                "domain": domain,
                "position": pos,
                "fetched_at": datetime.utcnow(),
            })
    return rows

def save_results(rows):
    if not rows:
        return
    with engine.begin() as conn:
        for r in rows:
            conn.execute(text("""
                INSERT INTO rank (keyword_id, domain, position, fetched_at)
                VALUES (:keyword_id, :domain, :position, :fetched_at)
            """), r)

# -------------------
# UI
# -------------------
st.title("🔎 SERP Tracker Dashboard")

# --- Keyword management ---
st.subheader("Manage Keywords")
new_kw = st.text_input("Add new keyword")
if st.button("Add Keyword"):
    if new_kw.strip():
        insert_keyword(new_kw.strip())
        st.success(f"Added keyword: {new_kw}")
        st.cache_data.clear()

keywords = load_keywords()
for _, row in keywords.iterrows():
    col1, col2 = st.columns([4,1])
    col1.write(row["text"])
    if col2.button("❌", key=f"del{row['id']}"):
        delete_keyword(row["id"])
        st.cache_data.clear()
        st.experimental_rerun()

# --- Fetch SERP ---
if st.button("Fetch SERP Data Now"):
    for _, row in keywords.iterrows():
        rows = fetch_serp_results(row["text"], row["id"])
        save_results(rows)
    st.success("SERP data fetched and saved.")
    st.cache_data.clear()

# --- Keyword selector ---
if not keywords.empty:
    sel = st.selectbox("Select a keyword", options=keywords["text"].tolist())
    kid = int(keywords[keywords["text"] == sel]["id"].iloc[0])
    df = fetch_rank_history(kid)

    if df.empty:
        st.info("No data yet. Fetch SERP data first.")
    else:
        view_mode = st.radio("View", ["Keyword trend", "Domain trends"])

        if view_mode == "Keyword trend":
            avg_rank = df.groupby("date")["position"].mean().reset_index()
            fig = px.line(avg_rank, x="date", y="position", title=f"Rank trend — {sel}")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            # anomaly detection
            if len(avg_rank) >= 14:
                series = avg_rank.set_index("date")["position"]
                stl = STL(series, period=7, robust=True).fit()
                resid = stl.resid
                med, mad = resid.median(), (resid - resid.median()).abs().median()
                denom = 1.4826 * mad if mad else resid.std()
                z = (resid - med) / (denom or 1)
                anomalies = z[abs(z) > 3.5]
                if not anomalies.empty:
                    st.markdown(f"**Anomalies detected:** {len(anomalies)}")
                    st.write(anomalies)

        elif view_mode == "Domain trends":
            fig = px.line(df, x="date", y="position", color="domain", title=f"Domain trends — {sel}")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

# --- Volatility Index ---
st.subheader("Volatility Index")
qry = """
SELECT fetched_at::date AS date, keyword_id, position
FROM rank
ORDER BY date
"""
with engine.connect() as conn:
    vdf = pd.read_sql(qry, conn)
if not vdf.empty:
    vdf = vdf.groupby(["keyword_id", "date"])["position"].mean().reset_index()
    vdf["change"] = vdf.groupby("keyword_id")["position"].diff().abs()
    vol = vdf.groupby("date")["change"].mean().reset_index()
    st.metric("Current Volatility", f"{vol['change'].iloc[-1]:.2f}")
    fig = px.line(vol, x="date", y="change", title="Volatility Index over time")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No volatility data yet. Fetch SERP data first.")
