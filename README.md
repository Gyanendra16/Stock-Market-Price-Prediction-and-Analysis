# Stock-Market-Price-Prediction-and-Analysis

# 📈 Stock Market Price Prediction and Analysis
---

## 🚀 How to Run

### Step 1 — Install Python Libraries
Open your terminal or command prompt and run:
```bash
pip install -r requirements.txt
```

### Step 2 — Start the Web App
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
stock_prediction/
│
├── app.py                     ← Main Streamlit web application
│
├── utils/
│   ├── data_loader.py         ← Downloads stock data + calculates indicators
│   └── preprocessor.py        ← Feature engineering + train/test split + scaling
│
├── models/
│   └── train_evaluate.py      ← Model training, prediction, evaluation
│
├── requirements.txt           ← Python package dependencies
└── README.md                  ← This file
```

---

## 🧠 How It Works (Step-by-Step)

| Step | Description |
|------|-------------|
| 1 | Download historical OHLCV data using **yfinance** |
| 2 | Calculate technical indicators: SMA (20, 50), EMA (12), RSI (14), MACD |
| 3 | Define Target: **next day's closing price** (Close shifted by -1) |
| 4 | Split data: **80% train / 20% test** (chronological, no shuffle) |
| 5 | Scale features using **StandardScaler** |
| 6 | Train 3 models: **Linear Regression, Decision Tree, Random Forest** |
| 7 | Evaluate using **R² Score, RMSE, MAE** |
| 8 | Display results with charts and predict the **next day's price** |

---

## 📊 Models Used

| Model | Library | Hyperparameters |
|-------|---------|-----------------|
| Linear Regression | `sklearn.linear_model` | Default |
| Decision Tree Regressor | `sklearn.tree` | `max_depth=5` |
| Random Forest Regressor | `sklearn.ensemble` | `n_estimators=100, random_state=42` |

---

## 📌 Indian Stock Ticker Examples

| Company | Ticker |
|---------|--------|
| TCS | `TCS.NS` |
| Reliance | `RELIANCE.NS` |
| Infosys | `INFY.NS` |
| HDFC Bank | `HDFCBANK.NS` |
| Wipro | `WIPRO.NS` |

---

## 📦 Python Libraries Used

- `yfinance` — Download stock data from Yahoo Finance
- `scikit-learn` — Machine learning models + preprocessing
- `pandas` — Data manipulation
- `numpy` — Numerical computations
- `matplotlib` — Charts and plots
- `streamlit` — Web interface

---

## 📚 References

1. Box, Jenkins, et al. — *Time Series Analysis: Forecasting and Control*, Wiley, 2015
2. Khare et al. — *Stock market prediction using linear regression and SVM*, IJAER, 2017
3. Breiman — *Random Forests*, Machine Learning Journal, 2001
4. Yahoo Finance API (yfinance) — https://pypi.org/project/yfinance/
