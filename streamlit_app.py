import os
import re
import time
import requests
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from api.config import settings
from datetime import datetime
import plotly.express as px
from statsmodels.tsa.seasonal import STL
from pytrends.request import TrendReq

# ----------------------
# Database setup
# ----------------------
def to_sync_url(async_url: str):
    return re.sub(r"\+asyncpg", "", async_url)

SYNC_DB_URL = to_sync_url(settings.DATABASE_URL)
engine = create_engine(SYNC_DB_URL, pool_pre_ping=True)

# ✅ check schema
from api.models import reset_db
inspector = inspect(engine)
existing_tables = inspector.get_table_names()
required_tables = {"keyword", "engine", "rank", "serp_feature", "anomaly"}
if not required_tables.issubset(set(existing_tables)):
    st.warning("⚠️ Database schema incomplete. Resetting...")
    reset_db()

SERP_API_KEY = settings.SERP_API_KEY

# ----------------------
# Database helpers
# ----------------------
def init_tables():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS keyword (
            id SERIAL PRIMARY KEY,
            text TEXT UNIQUE NOT NULL
        )
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS rank (
            id SERIAL PRIMARY KEY,
            keyword_id INTEGER REFERENCES keyword(id),
            domain TEXT NOT NULL,
            position INTEGER NOT NULL,
            fetched_at TIMESTAMP DEFAULT now()
        )
        """))

init_tables()

def load_keywords():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT id, text FROM keyword ORDER BY text", conn)
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

def save_results(rows):
    if not rows:
        return
    with engine.begin() as conn:
        for r in rows:
            conn.execute(text("""
                INSERT INTO rank (keyword_id, domain, position, fetched_at)
                VALUES (:keyword_id, :domain, :position, :fetched_at)
            """), r)

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
    df["date"] = pd.to_datetime(df["date"])
    return df

# ----------------------
# SERP fetch
# ----------------------
def fetch_serp_results(keyword: str, keyword_id: int, save_to_db=False):
    url = "https://serpapi.com/search"
    params = {
        "q": keyword,
        "engine": "google",
        "num": 100,  # ⚠️ note: SerpAPI may ignore this, but harmless
        "api_key": SERP_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        st.error(f"SERP API error: {e}")
        return []

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

    if save_to_db:
        save_results(rows)

    return rows

# ----------------------
# Google Trends volatility
# ----------------------
def get_volatility_from_trends(keywords, timeframe="now 7-d"):
    if not keywords:
        return pd.DataFrame()

    py = st.session_state.get("pytrends")
    if not py:
        py = TrendReq(hl="en-US", tz=360)
        st.session_state.pytrends = py

    all_series = []
    for kw in keywords:
        try:
            py.build_payload([kw], cat=0, timeframe=timeframe, geo="")
            df = py.interest_over_time()
            if df.empty:
                continue
            s = df[kw].pct_change().abs().rename(kw)
            all_series.append(s)
            time.sleep(0.5)  # avoid rate limits
        except Exception as e:
            st.warning(f"Trend fetch failed for {kw}: {e}")
            continue

    if not all_series:
        return pd.DataFrame()

    combined = pd.concat(all_series, axis=1)
    combined["volatility"] = combined.mean(axis=1)
    return combined.reset_index().rename(columns={"date": "date"})

# ----------------------
# Admin login
# ----------------------
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if not st.session_state.is_admin:
    with st.sidebar:
        pw = st.text_input("Admin password", type="password")
        if pw and pw == st.secrets.get("ADMIN_PASSWORD", ""):
            st.session_state.is_admin = True
            st.success("Admin mode enabled")
        elif pw:
            st.error("Wrong password")

if st.session_state.is_admin:
    st.sidebar.header("Admin / Options")
    save_to_db = st.sidebar.checkbox("Save fetched results to DB", value=False)
    show_db_keywords = st.sidebar.checkbox("Show persisted DB keywords", value=False)
else:
    save_to_db = False
    show_db_keywords = False

# ----------------------
# UI
# ----------------------
st.title("🔎 SERP Tracker Dashboard")

# Keyword management
st.subheader("Manage Keywords")
if "session_keywords" not in st.session_state:
    st.session_state.session_keywords = []

new_kw = st.text_input("Add new keyword (session only)")
if st.button("Add Keyword"):
    if new_kw.strip():
        st.session_state.session_keywords.append(new_kw.strip())
        st.success(f"Added keyword: {new_kw}")

# show persisted DB keywords only if admin enabled it
keywords = load_keywords() if show_db_keywords else pd.DataFrame(columns=["id", "text"])
session_kws = pd.DataFrame(
    [{"id": -(i+1), "text": kw} for i, kw in enumerate(st.session_state.session_keywords)]
)

all_keywords = pd.concat([session_kws, keywords], ignore_index=True)

for _, row in all_keywords.iterrows():
    col1, col2 = st.columns([4,1])
    col1.write(row["text"])
    if col2.button("❌", key=f"del{row['id']}"):
        if row["id"] < 0:  # session keyword
            st.session_state.session_keywords.remove(row["text"])
        else:
            delete_keyword(row["id"])
        st.experimental_rerun()

# Fetch SERP
if st.button("Fetch SERP Data Now"):
    for _, row in all_keywords.iterrows():
        rows = fetch_serp_results(row["text"], row["id"], save_to_db=save_to_db)
        if not save_to_db:
            st.session_state[f"serp_{row['text']}"] = pd.DataFrame(rows)
    st.success("SERP data fetched.")

# Keyword selector
if not all_keywords.empty:
    sel_kw = st.selectbox("Select a keyword", options=all_keywords["text"].tolist())
    kid = int(all_keywords[all_keywords["text"] == sel_kw]["id"].iloc[0])

    if kid < 0:  # session keyword
        df = st.session_state.get(f"serp_{sel_kw}", pd.DataFrame())
    else:  # persisted
        df = fetch_rank_history(kid)

    if df.empty:
        st.info("No data yet. Fetch SERP data first.")
    else:
        if "date" not in df.columns and "fetched_at" in df.columns:
            df["date"] = pd.to_datetime(df["fetched_at"])

        view_mode = st.radio("View", ["Keyword trend", "Domain trends"])

        if view_mode == "Keyword trend":
            avg_rank = df.groupby("date")["position"].mean().reset_index()
            fig = px.line(avg_rank, x="date", y="position", title=f"Rank trend — {sel_kw}")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            # anomaly detection
            if len(avg_rank) >= 14:
                from numpy import median
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
            df_snap = df.sort_values("position", ascending=True).head(9)
            fig = px.bar(
                df_snap.sort_values("position", ascending=True),
                x="position", y="domain",
                orientation="h", text="position",
                title=f"Top 9 domains for '{sel_kw}'"
            )
            fig.update_yaxes(autorange="reversed")  # ✅ ensures #1 at top
            st.plotly_chart(fig, use_container_width=True)

# Volatility Index
st.subheader("Volatility Index (Google Trends)")
kws = all_keywords["text"].tolist()
voldf = get_volatility_from_trends(kws, timeframe="now 7-d")
if not voldf.empty:
    st.metric("Current Volatility (Trends)", f"{voldf['volatility'].iloc[-1]:.3f}")
    st.plotly_chart(px.line(voldf, x="date", y="volatility", title="Volatility (Google Trends)"))
else:
    st.info("No volatility data available (Google Trends may be rate-limited).")
