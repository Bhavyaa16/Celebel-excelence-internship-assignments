"""
Retail Demand Forecasting Dashboard

Run with:  streamlit run dashboard/app.py
Requires artifacts/metrics.json (produced by src/evaluate.py) to exist.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.rag_assistant import ForecastRAG  # noqa: E402

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
HORIZON_OPTIONS = [7, 15, 30, 60, 90]

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")


@st.cache_data
def load_metrics():
    path = ARTIFACTS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


metrics = load_metrics()

st.sidebar.title("Demand Forecasting")
page = st.sidebar.radio(
    "View",
    ["Overview", "Series Explorer", "Model Performance", "Actionable Rollup", "Ask the Assistant"],
)

if page == "Overview":
    st.title("Multi-Series Forecasting for Retail Demand Optimization")
    st.caption("Celebal Technologies — CEI Data Science Internship")

    st.markdown(
        """
### Problem
Forecast daily demand across **500 store-item series** (10 stores x 50 items),
capturing both each series' own dynamics and patterns shared across all of
them, while scaling to many series and producing forecasts usable for
inventory and supply-chain decisions.
"""
    )

    st.subheader("Pipeline")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Data & Features**\n\nLags (1/7/14/28/365d), rolling "
                "mean/std, calendar features (day-of-week, month, weekend)")
    c2.markdown("**2. Baseline**\n\nNaive seasonal model — repeats last "
                "week forward, the sanity-check yardstick")
    c3.markdown("**3. Core Model**\n\nGlobal LSTM with store/item "
                "embeddings, trained across all 500 series at once")
    c4.markdown("**4. Serving Layer**\n\nStreamlit dashboard + RAG "
                "assistant (Groq) grounded in backtested results")

    st.subheader("Why a global LSTM, not 500 separate models")
    st.markdown(
        """
- **Series-specific dynamics** — the LSTM reads each series' own 90-day
  history, so it learns that series' own trend and recent pattern.
- **Shared temporal structure** — store and item IDs are fed in as learned
  embeddings, so the single model transfers seasonal/demand patterns
  learned from all 500 series, rather than each series learning in
  isolation from a short history.
- **Scalability** — one training run and one `predict()` call serves every
  series. Adding more stores/items doesn't mean training more models.
- **Multi-horizon** — the model outputs 90 days directly in one pass
  (not fed back recursively), so errors don't compound across the horizon.
"""
    )

    st.subheader("Tech stack")
    st.table(pd.DataFrame([
        {"Component": "Data pipeline", "Tech": "pandas, NumPy"},
        {"Component": "Core model", "Tech": "TensorFlow / Keras (LSTM + Embedding layers)"},
        {"Component": "Baseline", "Tech": "Naive seasonal (pure pandas)"},
        {"Component": "Evaluation", "Tech": "WAPE, RMSE — backtested on a held-out 3-month window"},
        {"Component": "RAG assistant", "Tech": "TF-IDF retrieval (scikit-learn) + Groq LLM (Llama 3.1)"},
        {"Component": "Dashboard", "Tech": "Streamlit, Plotly"},
    ]))

    st.subheader("Deliberately out of scope")
    st.markdown(
        "Transformers/attention, CNN-LSTM hybrids, GANs/VAEs, and agent "
        "frameworks were left out — not because they wouldn't work, but "
        "because they weren't needed to meet the brief, and adding them "
        "would trade a well-justified, explainable pipeline for "
        "complexity that doesn't add proportional value here."
    )
    st.stop()

if metrics is None:
    st.error(
        "No results found yet. Run `python -m src.train` then "
        "`python -m src.evaluate` to generate forecasts before launching the dashboard."
    )
    st.stop()

per_series_df = pd.DataFrame(
    [{"store": s["store"], "item": s["item"], "wape": s["wape"], "rmse": s["rmse"]}
     for s in metrics["per_series"]]
)


def get_series(store, item):
    for s in metrics["per_series"]:
        if str(s["store"]) == str(store) and str(s["item"]) == str(item):
            return s
    return None


if page == "Series Explorer":
    st.title("Series Explorer")
    col1, col2, col3 = st.columns(3)
    store = col1.selectbox("Store", sorted(per_series_df["store"].unique()))
    item = col2.selectbox("Item", sorted(per_series_df["item"].unique()))
    horizon = col3.selectbox("Forecast horizon (days)", HORIZON_OPTIONS, index=2)

    series = get_series(store, item)
    if series is None:
        st.warning("No backtest result for this store-item pair.")
    else:
        actual = series["actual"][:horizon]
        forecast = series["forecast"][:horizon]
        days = list(range(1, horizon + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=actual, name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=days, y=forecast, name="Forecast", mode="lines+markers"))
        fig.update_layout(
            title=f"Store {store} / Item {item} — {horizon}-day forecast vs actual (backtest)",
            xaxis_title="Day", yaxis_title="Units sold",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("WAPE", f"{series['wape']:.1%}")
        st.metric("RMSE", f"{series['rmse']:.2f}")

elif page == "Model Performance":
    st.title("Model Performance")
    st.metric("Overall WAPE", f"{metrics['overall']['wape']:.1%}")
    st.metric("Overall RMSE", f"{metrics['overall']['rmse']:.2f}")

    st.subheader("Error by store")
    st.bar_chart(per_series_df.groupby("store")["wape"].mean())

    st.subheader("Worst-performing series (highest WAPE)")
    st.dataframe(per_series_df.sort_values("wape", ascending=False).head(15))

elif page == "Actionable Rollup":
    st.title("Actionable Rollup")

    per_series_df["trend"] = [
        "up" if s["forecast"][-1] > s["forecast"][0] * 1.1
        else "down" if s["forecast"][-1] < s["forecast"][0] * 0.9
        else "stable"
        for s in metrics["per_series"]
    ]

    st.subheader("Trending up (potential restock candidates)")
    st.dataframe(per_series_df[per_series_df["trend"] == "up"].sort_values("wape"))

    st.subheader("Highest forecast uncertainty (safety stock candidates)")
    st.dataframe(per_series_df.sort_values("wape", ascending=False).head(10))

elif page == "Ask the Assistant":
    st.title("Ask the Assistant")
    st.caption("Answers are grounded in the model's backtested forecasts and metrics only.")

    question = st.text_input("Ask a question about the forecasts, e.g. "
                              "'Which items should I restock in store 4?'")
    if question:
        try:
            rag = ForecastRAG()
            with st.spinner("Thinking..."):
                answer = rag.answer(question)
            st.write(answer)
        except RuntimeError as e:
            st.error(str(e))
