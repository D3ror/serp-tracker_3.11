import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from api.config import settings
from datetime import timedelta
import plotly.express as px
from statsmodels.tsa.seasonal import STL
import numpy as np
import re

# Convert async URL to sync (postgresql+asyncpg:// -> postgresql://)
def to_sync_url(async_url: str):
    return re.sub(r"\+asyncpg", "", async_url)

SYNC_DB_URL = to_sync_url(settings.DATABASE_URL)
engine = create_engine(SYNC_DB_URL, pool_pre_ping=True)

@st.cache_data(ttl=600)
def load_keywords():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT id, text FROM keyword ORDER BY text", conn)
    return df

@st.cache_data
def fetch_rank_history(keyword_id: int, days=180):
    qry = text("""
       SELECT date, rank FROM rank WHERE keyword_id = :kid ORDER BY date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(qry, conn, params={"kid": keyword_id})
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').asfreq('D').interpolate()
    return df

st.title("SERP Tracker Dashboard")
keywords = load_keywords()
sel = st.selectbox("Keyword", options=keywords['text'].tolist())
if sel:
    kid = int(keywords[keywords['text'] == sel]['id'].iloc[0])
    df = fetch_rank_history(kid, days=365)
    if df.empty:
        st.info("No data yet.")
    else:
        df['rank'] = df['rank'].astype(float)
        fig = px.line(df, y='rank', title=f"Rank history — {sel}", labels={'index':'date','rank':'rank'})
        fig.update_yaxes(autorange="reversed")  # rank 1 is top, invert y-axis
        st.plotly_chart(fig, use_container_width=True)

        # STL decomposition and anomalies
        if len(df) >= 14:
            series = df['rank']
            stl = STL(series, period=7, robust=True).fit()
            comp = pd.DataFrame({
                'trend': stl.trend,
                'seasonal': stl.seasonal,
                'resid': stl.resid
            }, index=series.index)
            st.subheader("Decomposition")
            st.line_chart(comp)

            # robust z-score on residuals
            med = comp['resid'].median()
            mad = (comp['resid'] - med).abs().median()
            denom = 1.4826 * mad if mad != 0 else comp['resid'].std()
            z = (comp['resid'] - med) / (denom or 1.0)
            anomalies = z[abs(z) > 3.5]
            if not anomalies.empty:
                st.markdown(f"**Anomalies**: {len(anomalies)}")
                st.write(anomalies.to_frame("zscore"))

                # overlay anomalies on the main chart
                fig.add_scatter(x=anomalies.index, y=series.loc[anomalies.index],
                                mode='markers', marker=dict(size=8, symbol='x'),
                                name='anomaly')
                st.plotly_chart(fig, use_container_width=True)