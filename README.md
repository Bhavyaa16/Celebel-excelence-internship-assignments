# Multi-Series Forecasting for Retail Demand Optimization

Celebal Technologies — CEI Data Science Internship project.

Forecasts daily demand across 500 store-item series (10 stores x 50 items)
using a single global LSTM model, then serves the results through a
Streamlit dashboard with a RAG-based Q&A assistant on top.

## Why this approach

- **One global LSTM/GRU**, not 500 separate models — store and item are fed
  in as learned embeddings, so the model shares seasonal/trend patterns
  across all series while still telling them apart. This scales to many
  series far better than per-series classical models, and directly captures
  the sequence dynamics (trend, autocorrelation, seasonality) the problem
  calls for.
- **Naive seasonal baseline** included as a sanity check — if the LSTM can't
  beat "repeat last week," that's a real problem worth catching early.
- **RAG layer** (TF-IDF retrieval + Groq LLM) lets a non-technical user ask
  plain-English questions about the forecasts, grounded strictly in the
  model's own backtested results — not a general chatbot bolted onto the
  side.

## Project structure

```
data/                   put train.csv here (see data/README.md)
src/
  data_loader.py        loads + validates the raw sales data
  feature_engineering.py lag/rolling/calendar features
  sequence_builder.py   builds sliding-window sequences for the RNN
  models/
    baseline.py          naive seasonal forecast
    lstm_gru.py          the global LSTM/GRU architecture
  train.py               trains the model, saves artifacts/lstm_model.keras
  evaluate.py            backtests the model, saves artifacts/metrics.json
  rag_assistant.py        RAG Q&A over the backtested results
dashboard/
  app.py                  Streamlit dashboard (4 pages)
artifacts/                generated: trained model, metrics, id maps
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

1. Download `train.csv` from Kaggle (link in `data/README.md`) and place it
   in `data/train.csv`.
2. Get a free Groq API key at https://console.groq.com/keys, copy
   `.env.example` to `.env`, and paste it in.

## Running the pipeline

```bash
# 1. Train the model (this is the slow step — builds ~hundreds of thousands
#    of training windows, then trains the LSTM)
python -m src.train

# 2. Backtest and generate the metrics/forecasts the dashboard reads
python -m src.evaluate

# 3. Launch the dashboard
streamlit run dashboard/app.py
```

## Dashboard pages

1. **Overview** — project goal, pipeline, model justification, and tech
   stack per component, for anyone new to the project.
2. **Series Explorer** — pick a store/item, view actual vs forecasted
   demand at a chosen horizon (7/15/30/60/90 days), with backtest error.
3. **Model Performance** — overall WAPE/RMSE, error broken down by store,
   worst-performing series.
4. **Actionable Rollup** — series trending up (restock candidates) and
   series with the highest forecast error (safety-stock candidates).
5. **Ask the Assistant** — natural-language Q&A grounded in the backtested
   forecasts and metrics only.

## Notes on scope

This project intentionally stays within a defined set of techniques
(RNN/LSTM/GRU sequence modeling, embeddings, standard optimizers, and a
lightweight RAG layer) rather than adding architectures like Transformers,
GANs, or agent frameworks that weren't part of the intended scope. The goal
is a well-justified, explainable pipeline rather than the most complex one.
