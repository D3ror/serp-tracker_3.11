import os
import re
import time
import requests
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from api.config import settings
from api.models import reset_db
from datetime import datetime, timedelta
import plotly.express as px
from statsmodels.tsa.seasonal import STL
import numpy as np
from pytrends.request import TrendReq

# ---------------------
# Database / engine
# ---------------------
def to_sync_url(async_url: str):
    return re.sub(r"\+asyncpg", "", async_url)

SYNC_DB_URL = to_sync_url(settings.DATABASE_URL)
engine = create_engine(SYNC_DB_URL, pool_pre_ping=True)

# ---------------------
# Safety: ensure schema has 'domain' column (non-destructive)
# ---------------------
try:
    inspector = inspect(engine)
    if "rank" in inspector.get_table_names():
        cols = [c["name"] for c in inspector.get_columns("rank")]
        if "domain" not in cols:
            # attempt to add domain column non-destructively
            with engine.begin() as conn:
                try:
                    conn.execute(text("ALTER TABLE rank ADD COLUMN IF NOT EXISTS domain TEXT"))
                    conn.execute(text("UPDATE rank SET domain = 'unknown' WHERE domain IS NULL"))
                    # clear DB caches (if any)
                except Exception as exc:
                    # If ALTER fails, print to logs but continue (we will show helpful errors later)
                    st.warning("Could not add 'domain' column automatically: " + str(exc))
    else:
        # If rank table missing, create full schema (calls models.reset_db, which drops+creates)
        # For portfolio/demo it's okay to create.
        reset_db()
except Exception as e:
    st.warning("DB schema check failed: " + str(e))

# ---------------------
# Session state setup
# ---------------------
if "session_keywords" not in st.session_state:
    st.session_state.session_keywords = []     # list of strings (ephemeral)
if "last_results" not in st.session_state:
    st.session_state.last_results = {}        # mapping keyword -> list[dict]
if "pytrends" not in st.session_state:
    # reuse TrendReq instance per session to avoid repeated auth overhead
    try:
        st.session_state.pytrends = TrendReq(hl="en-US", tz=360)
    except Exception:
        st.session_state.pytrends = None

# ---------------------
# Helpers: DB operations (admin opt-in only)
# ---------------------
def get_or_create_keyword_row(keyword_text: str):
    """
    Ensure keyword exists in keyword table and return its id.
    """
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO keyword (text) VALUES (:t) ON CONFLICT (text) DO NOTHING"), {"t": keyword_text})
        row = conn.execute(text("SELECT id FROM keyword WHERE text = :t"), {"t": keyword_text}).fetchone()
        return int(row[0]) if row else None

def save_rows_to_db(rows):
    """
    rows: iterable of dicts with keys keyword_id, domain, position, fetched_at
    """
    if not rows:
        return
    with engine.begin() as conn:
        for r in rows:
            conn.execute(
                text("""
                    INSERT INTO rank (keyword_id, domain, position, fetched_at)
                    VALUES (:keyword_id, :domain, :position, :fetched_at)
                """),
                r,
            )

def delete_keyword_from_db_by_id(kw_id: int):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM rank WHERE keyword_id = :id"), {"id": kw_id})
        conn.execute(text("DELETE FROM keyword WHERE id = :id"), {"id": kw_id})

# ---------------------
# SERP fetching (session-only)
# ---------------------
SERP_API_KEY = settings.SERP_API_KEY or os.getenv("SERP_API_KEY")

def fetch_serp_session(keyword: str, max_results: int = 100, sleep_s: float = 0.5):
    """
    Fetch SERP results from SerpAPI (session-only). Returns list of dicts:
      {"domain": ..., "position": int, "link": ..., "fetched_at": datetime}
    Does NOT write to DB.
    """
    if not SERP_API_KEY:
        st.error("SERP_API_KEY is not set. Set it in Streamlit Secrets to fetch live SERPs.")
        return []

    url = "https://serpapi.com/search"
    params = {
        "q": keyword,
        "engine": "google",
        "num": max_results,
        "api_key": SERP_API_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
    except requests.RequestException as e:
        st.error(f"Network error when fetching SERP for '{keyword}': {e}")
        return []

    if r.status_code != 200:
        st.error(f"SerpAPI returned status {r.status_code} for '{keyword}': {r.text[:200]}")
        return []

    try:
        data = r.json()
    except ValueError:
        st.error("SerpAPI returned non-JSON data.")
        return []

    # rate limit friendly pause
    time.sleep(sleep_s)

    organic = data.get("organic_results") or data.get("organic_results", [])
    results = []
    for res in organic:
        pos = res.get("position")
        link = res.get("link") or res.get("url") or ""
        domain = re.sub(r"^https?://(www\.)?", "", link).split("/")[0] if link else "unknown"
        if pos is None:
            # Some results might not include explicit 'position', skip them
            continue
        results.append({
            "domain": domain,
            "position": int(pos),
            "link": link,
            "fetched_at": datetime.utcnow(),
        })
    return results

# ---------------------
# Google Trends helpers
# ---------------------
def get_trends_interest(keyword: str, timeframe: str = "now 7-d"):
    """
    Return a DataFrame with columns ['date', 'interest'] or empty DF if failed.
    """
    py = st.session_state.get("pytrends")
    if not py:
        try:
            py = TrendReq(hl="en-US", tz=360)
            st.session_state.pytrends = py
        except Exception as e:
            st.warning(f"Failed to initialize pytrends: {e}")
            return pd.DataFrame()
    try:
        py.build_payload([keyword], cat=0, timeframe=timeframe, geo="")
        df = py.interest_over_time()
        if df.empty:
            return pd.DataFrame()
        # select column and drop isPartial if exists
        series = df[keyword].rename("interest").reset_index()
        series = series[["date", "interest"]]
        return series
    except Exception as e:
        # often blocked, quota problem, or network issue
        st.warning(f"Google Trends failed for '{keyword}': {e}")
        return pd.DataFrame()

def get_volatility_from_trends(keywords, timeframe="now 7-d"):
    """
    For each keyword, compute day-to-day absolute change in interest series.
    Aggregate by date across keywords and return DataFrame(date, volatility)
    """
    if not keywords:
        return pd.DataFrame()
    per_kw = []
    for kw in keywords:
        tdf = get_trends_interest(kw, timeframe=timeframe)
        if tdf.empty:
            continue
        tdf["change"] = tdf["interest"].diff().abs().fillna(0)
        per_kw.append(tdf[["date", "change"]].set_index("date")["change"])
    if not per_kw:
        return pd.DataFrame()
    # combine by outer join and take mean across keywords per day
    combined = pd.concat(per_kw, axis=1).fillna(0)
    combined["vol"] = combined.mean(axis=1)
    vol_df = combined[["vol"]].reset_index().rename(columns={"vol": "volatility"})
    return vol_df

# ---------------------
# UI layout: controls & session keywords
# ---------------------
st.set_page_config(page_title="SERP Tracker (Demo)", layout="wide")
st.title("🔎 SERP Tracker — privacy-first demo")

# sidebar admin controls
st.sidebar.header("Admin / Options")
save_to_db = st.sidebar.checkbox("Save fetched results to DB (admin only)", value=False)
show_db_keywords = st.sidebar.checkbox("Show persisted DB keywords (admin)", value=False)

st.sidebar.markdown("**Notes**")
st.sidebar.write("- Keywords are session-only by default (clean slate per visitor).")
st.sidebar.write("- Toggle 'Save fetched results to DB' to persist (admin).")

# Session keyword input
st.subheader("Session Keywords (private to you)")
col_a, col_b = st.columns([4,1])
with col_a:
    kw_in = st.text_input("Add a keyword to your session (press Enter first)")
with col_b:
    if st.button("Add to session"):
        if kw_in and kw_in.strip():
            st.session_state.session_keywords.append(kw_in.strip())
            # immediate feedback
            st.success(f"Added '{kw_in.strip()}' to your session")
            # rerun to display updated list
            if hasattr(st, "experimental_rerun"):
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

# show session keywords with ability to remove
if st.session_state.session_keywords:
    st.write("### Your session keywords")
    for i, k in enumerate(list(st.session_state.session_keywords)):
        c1, c2 = st.columns([8,1])
        c1.write(f"{i+1}. {k}")
        if c2.button("Remove", key=f"rm_{i}"):
            st.session_state.session_keywords.pop(i)
            if hasattr(st, "experimental_rerun"):
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
else:
    st.info("No session keywords yet. Add one above.")

# show persisted keywords (admin)
if show_db_keywords:
    try:
        with engine.connect() as conn:
            dbk = pd.read_sql("SELECT id, text FROM keyword ORDER BY text", conn)
        st.write("### Persisted keywords (DB)")
        st.table(dbk)
        # allow admin delete
        st.write("Delete a persisted keyword (admin):")
        if not dbk.empty:
            del_id = st.number_input("Keyword ID to delete", min_value=int(dbk["id"].min()), max_value=int(dbk["id"].max()), value=int(dbk["id"].min()))
            if st.button("Delete persisted keyword"):
                delete_keyword_from_db_by_id(int(del_id))
                st.success(f"Deleted persisted keyword id={del_id}")
                # clear any relevant caches
                if hasattr(st, "experimental_rerun"):
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
    except Exception as e:
        st.warning("Could not load persisted keywords: " + str(e))

# ---------------------
# Fetch session-only (and optionally save)
# ---------------------
st.markdown("---")
st.subheader("Fetch SERP (session-only)")

if st.button("Fetch SERP Data Now (session-only)"):
    if not st.session_state.session_keywords:
        st.warning("No session keywords to fetch.")
    else:
        all_rows_to_save = []  # aggregated rows to optionally save to DB (with keyword_id)
        for kw in st.session_state.session_keywords:
            st.info(f"Fetching SERP for: {kw}")
            rows = fetch_serp_session(kw)
            # store results in session under keyword
            st.session_state.last_results[kw] = rows
            # if admin asked to persist, map keyword -> id and prepare rows to write
            if save_to_db:
                kw_id = get_or_create_keyword_row(kw)
                # map rows into DB format
                dbrows = []
                for r in rows:
                    dbrows.append({
                        "keyword_id": kw_id,
                        "domain": r["domain"],
                        "position": int(r["position"]),
                        "fetched_at": r["fetched_at"],
                    })
                all_rows_to_save.extend(dbrows)
            # be nice to the API
        if save_to_db and all_rows_to_save:
            try:
                save_rows_to_db(all_rows_to_save)
                st.success("Results saved to DB.")
            except Exception as e:
                st.error("Failed to save results to DB: " + str(e))
        else:
            st.info("Results kept session-only (not saved).")
        # For privacy: clear session keywords after fetch if not saving
        if not save_to_db:
            st.session_state.session_keywords = []
        # clear caches used for DB reads so admin views update
        try:
            st.cache_data.clear()
        except Exception:
            pass
        if hasattr(st, "experimental_rerun"):
            try:
                st.experimental_rerun()
            except Exception:
                pass

# ---------------------
# Show results for a selected session keyword
# ---------------------
st.markdown("---")
st.subheader("Results & Analytics (session)")

if st.session_state.last_results:
    sel_kw = st.selectbox("Select a session keyword with results", options=list(st.session_state.last_results.keys()))
    selected_rows = st.session_state.last_results.get(sel_kw, [])
    if not selected_rows:
        st.info("No SERP rows for this session keyword. Run fetch.")
    else:
        # Top-9 domain snapshot
        st.write("### Top 9 domains (current snapshot)")
        df_snap = pd.DataFrame(selected_rows)
        # Ensure positions exist and are ints
        if "position" in df_snap.columns:
            df_snap = df_snap.sort_values("position").head(9)
            # horizontal bar - invert x axis so pos 1 appears at left
            fig = px.bar(df_snap, x="position", y="domain", orientation="h", text="position",
                         title=f"Top 9 domains for '{sel_kw}'")
            fig.update_layout(yaxis={"categoryorder":"total ascending"})
            fig.update_xaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.table(df_snap.head(9))

        # Domain table with link
        st.write("### Raw SERP rows")
        st.table(df_snap[["position", "domain", "link", "fetched_at"]].reset_index(drop=True))

        # Keyword Trend (Google Trends)
        st.write("### Keyword Trend (Google Trends, last 7 days)")
        trend_df = get_trends_interest(sel_kw, timeframe="now 7-d")
        if not trend_df.empty:
            fig2 = px.line(trend_df, x="date", y="interest", title=f"Search interest for '{sel_kw}' (7d)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No Google Trends data for this keyword (pytrends may be rate-limited).")

else:
    st.info("No fetched session results yet. Use 'Fetch SERP Data Now' to populate session results.")

# ---------------------
# Global Volatility Index (from Google Trends across session keywords)
# ---------------------
st.markdown("---")
st.subheader("Volatility Index (from Google Trends)")

vol_window = st.selectbox("Volatility window (days)", [3,7], index=0)

# Use persisted DB keywords only if admin opted to save and asked to compute from DB
if save_to_db and st.sidebar.checkbox("Compute volatility from persisted DB instead of Google Trends", value=False):
    # compute from DB if enough history
    try:
        # require at least 2 distinct dates in DB to compute changes
        with engine.connect() as conn:
            vdf = pd.read_sql("""
                SELECT fetched_at::date AS date, keyword_id, AVG(position) AS position
                FROM rank
                WHERE fetched_at >= (current_date - :wnd::int)
                GROUP BY keyword_id, date
                ORDER BY date
            """, conn, params={"wnd": vol_window})
        if vdf.empty or vdf["date"].nunique() < 2:
            st.info("Not enough persisted history in DB to compute volatility.")
        else:
            vdf["date"] = pd.to_datetime(vdf["date"])
            vdf["change"] = vdf.groupby("keyword_id")["position"].diff().abs()
            vol = vdf.groupby("date")["change"].mean().reset_index()
            st.metric("Current Volatility (DB)", f"{vol['change'].iloc[-1]:.2f}")
            figv = px.line(vol, x="date", y="change", title="Volatility Index (DB-based)")
            st.plotly_chart(figv, use_container_width=True)
    except Exception as e:
        st.error("Failed to compute DB volatility: " + str(e))
else:
    # Use Google Trends across current session keywords (if any)
    kws = st.session_state.session_keywords.copy()
    if not kws:
        # If session keywords empty, optionally fall back to persisted DB keywords for admin
        if save_to_db:
            try:
                with engine.connect() as conn:
                    dbk = pd.read_sql("SELECT id, text FROM keyword ORDER BY text", conn)
                kws = dbk["text"].tolist()[:10]  # limit number of trend fetches
            except Exception:
                kws = []
    if not kws:
        st.info("No keywords available for volatility computation. Add session keywords or save keywords to DB.")
    else:
        # get volatility series aggregated from Google Trends
        try:
            voldf = get_volatility_from_trends(kws, timeframe=f"now {vol_window}-d")
            if voldf.empty:
                st.info("No Google Trends volatility data available (API may be rate-limited).")
            else:
                st.metric("Current Volatility (Trends)", f"{voldf['volatility'].iloc[-1]:.2f}")
                st.plotly_chart(px.line(voldf, x="date", y="volatility", title="Volatility (from Google Trends)"), use_container_width=True)
        except Exception as e:
            st.warning("Computing volatility from Trends failed: " + str(e))

# ---------------------
# Footer: privacy / explanation
# ---------------------
st.markdown("---")
st.caption("This demo keeps keywords session-local by default so each visitor has a clean slate. Toggle 'Save fetched results to DB' in the sidebar to persist results (admin only). Google Trends is used for trend/volatility when there isn't enough stored SERP history.")
