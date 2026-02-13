# Narsingdi Air Quality Web App (AQI + PM2.5)

This is a defense-ready demo web app for **monitoring and predicting**:
- **Numeric AQI + AQI category**
- **PM2.5 (µg/m³)**

It follows the accepted paper's approach: **stacked ensemble (RF + XGBoost + LightGBM) with Ridge meta-learner**.

## 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Train models (creates `models/`)
```bash
python train_model.py --csv data/Narsingdi(BD)_AQI_2020_2024.csv --target AQI
python train_model.py --csv data/Narsingdi(BD)_AQI_2020_2024.csv --target PM2.5
```

## 3) Run the web app
```bash
python server.py
```
Open http://127.0.0.1:5000

## Notes
- Prediction page uses **history-based engineered features** (lags + rolling means).
- AQI category is derived from numeric AQI using standard AQI bands.

