# ui/streamlit_app.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
import numpy as np

# ----------------------
# Safe rerun helper (works across Streamlit versions)
# ----------------------
def safe_rerun():
    """
    Try to rerun the script. If st.experimental_rerun is missing or fails,
    toggle a session key and stop the script so the UI refreshes cleanly.
    """
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass
    # fallback: toggle a key and stop execution (forces front-end refresh)
    st.session_state["_rerun"] = not st.session_state.get("_rerun", False)
    st.stop()

# ----------------------
# Database setup
# ----------------------
def to_sync_url(async_url: str):
    return re.sub(r"\+asyncpg", "", async_url)

SYNC_DB_URL = to_sync_url(settings.DATABASE_URL)
engine = create_engine(SYNC_DB_URL, pool_pre_ping=True)

# ensure schema present (models.reset_db will drop & recreate)
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
# SERP fetch (session-first)
# ----------------------
def fetch_serp_results(keyword: str, keyword_id: int, save_to_db=False):
    """
    Fetch SERP from SerpAPI. Returns list of dicts (not persisted unless save_to_db=True).
    """
    if not SERP_API_KEY:
        st.error("SERP_API_KEY not configured. Set it in Streamlit Secrets for live SERP.")
        return []
    url = "https://serpapi.com/search"
    params = {
        "q": keyword,
        "engine": "google",
        "num": 100,
        "api_key": SERP_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
    except requests.RequestException as e:
        st.error(f"SERP API error for '{keyword}': {e}")
        return []

    try:
        data = r.json()
    except Exception:
        st.error("SerpAPI returned unexpected response.")
        return []

    organic = data.get("organic_results") or []
    rows = []
    for res in organic:
        pos = res.get("position")
        link = res.get("link", "") or res.get("url", "")
        domain = re.sub(r"^https?://(www\.)?", "", link).split("/")[0] if link else "unknown"
        if pos is None:
            continue
        rows.append({
            "keyword_id": keyword_id,
            "domain": domain,
            "position": int(pos),
            "fetched_at": datetime.utcnow(),
        })

    if save_to_db and rows:
        try:
            save_results(rows)
        except Exception as e:
            st.error(f"Failed to save SERP rows to DB: {e}")

    return rows

# ----------------------
# Google Trends volatility (aggregate pct-change)
# ----------------------
def get_volatility_from_trends(keywords, timeframe="now 7-d"):
    if not keywords:
        return pd.DataFrame()

    py = st.session_state.get("pytrends")
    if not py:
        try:
            py = TrendReq(hl="en-US", tz=360)
            st.session_state.pytrends = py
        except Exception as e:
            st.warning(f"Failed to initialize Google Trends client: {e}")
            return pd.DataFrame()

    all_series = []
    for kw in keywords:
        if not kw or pd.isna(kw):
            continue
        try:
            py.build_payload([kw], cat=0, timeframe=timeframe, geo="")
            df = py.interest_over_time()
            if df.empty:
                continue
            # interest_over_time returns timestamp index; compute abs pct change
            s = df[kw].pct_change().abs().rename(kw)
            s.index = pd.to_datetime(s.index)
            all_series.append(s)
            time.sleep(0.5)
        except Exception as e:
            # non-fatal: continue with others, but warn
            st.warning(f"Google Trends failed for '{kw}': {e}")
            continue

    if not all_series:
        return pd.DataFrame()

    combined = pd.concat(all_series, axis=1).fillna(method="ffill").fillna(0)
    combined["volatility"] = combined.mean(axis=1)
    out = combined[["volatility"]].reset_index().rename(columns={"index": "date"})
    # clean and sort
    out = out.dropna(subset=["volatility"]).sort_values("date")
    return out

# ----------------------
# Admin login (hidden options)
# ----------------------
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if not st.session_state.is_admin:
    with st.sidebar:
        pw = st.text_input("Admin password", type="password")
        if pw:
            if pw == st.secrets.get("ADMIN_PASSWORD", ""):
                st.session_state.is_admin = True
                st.success("Admin mode enabled")
                safe_rerun()
            else:
                st.error("Wrong password")

if st.session_state.is_admin:
    st.sidebar.header("Admin / Options")
    save_to_db = st.sidebar.checkbox("Save fetched results to DB", value=False)
    show_db_keywords = st.sidebar.checkbox("Show persisted DB keywords", value=False)
else:
    save_to_db = False
    show_db_keywords = False

# ----------------------
# UI: header & notes
# ----------------------
st.title("SERP tracker dashboard")
st.text("This web application collects SERP snapshots (public web results). Each visitor uses a session-first experience by default.")

# ----------------------
# Keyword management (session-only by default)
# ----------------------
st.subheader("Manage keywords")
if "session_keywords" not in st.session_state:
    st.session_state.session_keywords = []

new_kw = st.text_input("Add new keyword (session only)")
if st.button("Add keyword"):
    if new_kw and new_kw.strip():
        st.session_state.session_keywords.append(new_kw.strip())
        st.success(f"Added keyword: {new_kw.strip()}")
        safe_rerun()

# show persisted DB keywords only if admin enabled it
keywords_df = load_keywords() if show_db_keywords else pd.DataFrame(columns=["id", "text"])
session_kws = pd.DataFrame([{"id": -(i+1), "text": kw} for i, kw in enumerate(st.session_state.session_keywords)])
all_keywords = pd.concat([session_kws, keywords_df], ignore_index=True, sort=False)

# render keywords and deletion buttons
for _, row in all_keywords.iterrows():
    col1, col2 = st.columns([4, 1])
    col1.write(row["text"])
    if col2.button("❌", key=f"del{row['id']}"):
        if int(row["id"]) < 0:
            # session keyword
            try:
                st.session_state.session_keywords.remove(row["text"])
            except ValueError:
                pass
        else:
            # persisted
            delete_keyword(int(row["id"]))
        safe_rerun()

# ----------------------
# Fetch SERP (session-first)
# ----------------------
if st.button("Fetch SERP data"):
    if all_keywords.empty:
        st.info("No keywords available to fetch. Add one.")
    else:
        for _, row in all_keywords.iterrows():
            kw_text = row["text"]
            kw_id = int(row["id"])
            rows = fetch_serp_results(kw_text, kw_id, save_to_db=save_to_db)
            if not save_to_db:
                # store session DataFrame for UI access
                st.session_state[f"serp_{kw_text}"] = pd.DataFrame(rows)
        st.success("SERP data fetched.")
        safe_rerun()

# ----------------------
# Keyword selector & Views
# ----------------------
if not all_keywords.empty:
    sel_kw = st.selectbox("Select a keyword", options=all_keywords["text"].tolist())
    kid = int(all_keywords[all_keywords["text"] == sel_kw]["id"].iloc[0])

    # load dataframe: session snapshot or DB history
    if kid < 0:
        df = st.session_state.get(f"serp_{sel_kw}", pd.DataFrame())
        # if stored as list of dicts, convert
        if isinstance(df, list):
            df = pd.DataFrame(df)
    else:
        df = fetch_rank_history(kid)

    if df.empty:
        st.info("No data yet. Fetch SERP data first.")
    else:
        # ensure date column present
        if "date" not in df.columns and "fetched_at" in df.columns:
            df["date"] = pd.to_datetime(df["fetched_at"])

        view_mode = st.radio("View", ["Keyword trend", "Domain trends"])

        # ----------------------
        # KEYWORD TREND: domain vs position snapshot + per-domain anomaly detection
        # ----------------------
        if view_mode == "Keyword trend":
            # Snapshot: last known position per domain
            try:
                last_by_domain = (df.sort_values("date")
                                    .groupby("domain", as_index=False)
                                    .last()[["domain", "position", "date"]]
                                    .sort_values("position", ascending=True))
            except Exception:
                last_by_domain = df.sort_values("position").drop_duplicates(subset=["domain"])[["domain", "position", "date"]]

            fig = px.bar(
                last_by_domain,
                x="position", y="domain",
                orientation="h", text="position",
                title=f"Keyword trend — {sel_kw}"
            )
            # ensure 1 appears at top visually
            fig.update_yaxes(categoryorder="array", categoryarray=last_by_domain["domain"].tolist()[::-1])
            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            # Per-domain anomaly detection (STL) when we have enough history for that domain
            st.subheader("Per-domain anomaly detection")
            anomalies_summary = []
            domains = df["domain"].unique().tolist()
            for dom in domains:
                dom_df = df[df["domain"] == dom].sort_values("date")
                if len(dom_df) < 14:
                    # not enough points for STL — skip but compute simple changes if desired
                    continue
                # build a daily series (fill missing days by forward fill)
                ser = dom_df.set_index("date").resample("D")["position"].mean().interpolate()
                try:
                    stl = STL(ser, period=7, robust=True).fit()
                    resid = stl.resid
                    med = resid.median()
                    mad = (resid - med).abs().median()
                    denom = 1.4826 * mad if mad != 0 else resid.std()
                    z = (resid - med) / (denom or 1.0)
                    anoms = z[abs(z) > 3.5]
                    if not anoms.empty:
                        for idx, val in anoms.items():
                            anomalies_summary.append({"domain": dom, "date": idx.date(), "zscore": float(val)})
                except Exception as e:
                    # If STL fails for this domain, skip and continue quietly
                    continue

            if anomalies_summary:
                ast = pd.DataFrame(anomalies_summary).sort_values(["date", "zscore"], ascending=[False, False])
                st.write("Anomalies detected (domain, date, z-score):")
                st.table(ast)
            else:
                st.info("No domain-level anomalies detected (not enough history or no spikes).")

        # ----------------------
        # DOMAIN TRENDS: top 9 placement distribution (snapshot)
        # ----------------------
        elif view_mode == "Domain trends":
            df_snap = df.sort_values("position", ascending=True).head(9)
            # ensure we have 'position' ints and domain strings
            df_snap = df_snap.assign(position=df_snap["position"].astype(int))
            fig = px.bar(
                df_snap.sort_values("position", ascending=True),
                x="position", y="domain",
                orientation="h", text="position",
                title="Top 9 placement distribution"
            )
            # show #1 at top
            fig.update_yaxes(categoryorder="array", categoryarray=df_snap["domain"].tolist()[::-1])
            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

# ----------------------
# Volatility Index (Google Trends)
# ----------------------
st.subheader("Search volume volatility (Google Trends)")
kws = all_keywords["text"].tolist()
try:
    voldf = get_volatility_from_trends(kws, timeframe="now 7-d")
except NameError:
    # compatibility: if function named differently, try the alternative name
    voldf = pd.DataFrame()

if not voldf.empty:
    # drop NaNs and sort
    voldf = voldf.dropna(subset=["volatility"]).sort_values("date")
    st.metric("Current volatility (Trends)", f"{voldf['volatility'].iloc[-1]:.3f}")
    st.plotly_chart(px.line(voldf, x="date", y="volatility", title="Volatility (Google Trends)"), use_container_width=True)
else:
    st.info("No volatility data available (Google Trends may be rate-limited or returned no results).")
