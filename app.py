import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from utils.data_loader import download_stock_data, add_technical_indicators
from utils.preprocessor import prepare_features, split_and_scale
from models.train_evaluate import train_all_models, evaluate_model, predict_next_day


st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.main-title {
    font-size: 2.2rem; font-weight: 700;
    text-align: center; margin-bottom: 0.2rem;
}
.metric-card {
    background: #1e293b; border-radius: 10px;
    padding: 1rem; color: white; text-align: center; margin-bottom: 1rem;
}
.metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.75; }
.metric-card p  { margin: 0; font-size: 1.7rem; font-weight: 700; }
.wl-title {
    font-size: 0.78rem; font-weight: 700; color: #94a3b8;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px;
}
.wl-item {
    display: flex; justify-content: space-between; align-items: center;
    background: #1e293b; border-radius: 7px;
    padding: 6px 10px; margin-bottom: 5px; font-size: 0.82rem;
}
.wl-sym   { font-weight: 600; color: #e2e8f0; }
.wl-price { color: #e2e8f0; }
.wl-up    { color: #22c55e; font-size: 0.75rem; }
.wl-dn    { color: #ef4444; font-size: 0.75rem; }
.week52-wrap { margin: 4px 0 10px; }
.week52-label {
    display: flex; justify-content: space-between;
    font-size: 0.72rem; color: #64748b; margin-bottom: 3px;
}
.week52-bar {
    height: 5px; border-radius: 3px;
    background: linear-gradient(to right, #ef4444, #fbbf24, #22c55e);
    position: relative;
}
.week52-dot {
    position: absolute; top: -3px;
    width: 11px; height: 11px;
    background: white; border: 2px solid #3b82f6;
    border-radius: 50%; transform: translateX(-50%);
}
.footer {
    text-align: center; color: #888; font-size: 0.82rem;
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "AAPL"]


@st.cache_data(ttl=300)
def quick_quote(sym):
    try:
        d = yf.download(sym, period="2d", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if len(d) < 2:
            return None, None
        price = float(d["Close"].iloc[-1])
        prev  = float(d["Close"].iloc[-2])
        pct   = (price - prev) / prev * 100
        return price, pct
    except:
        return None, None


@st.cache_data(ttl=3600)
def week52_data(sym):
    try:
        d = yf.download(sym, period="1y", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if d.empty:
            return None, None, None
        lo  = float(d["Low"].min())
        hi  = float(d["High"].max())
        cur = float(d["Close"].iloc[-1])
        return lo, hi, cur
    except:
        return None, None, None


with st.sidebar:

    st.markdown('<div class="wl-title">⭐ Watchlist</div>', unsafe_allow_html=True)

    new_sym = st.text_input("", placeholder="Add ticker e.g. MSFT",
                            label_visibility="collapsed", key="wl_input")
    if st.button("＋ Add to Watchlist", use_container_width=True, key="wl_add"):
        sym = new_sym.strip().upper()
        if sym and sym not in st.session_state.watchlist:
            st.session_state.watchlist.append(sym)
            st.rerun()

    to_remove = None
    for sym in st.session_state.watchlist:
        price, pct = quick_quote(sym)
        if price is None:
            continue
        arrow = "▲" if pct >= 0 else "▼"
        cls   = "wl-up" if pct >= 0 else "wl-dn"
        ca, cb = st.columns([5, 1])
        with ca:
            st.markdown(f"""
            <div class="wl-item">
                <span class="wl-sym">{sym}</span>
                <span>
                    <span class="wl-price">{price:,.1f}</span>
                    <span class="{cls}"> {arrow}{abs(pct):.1f}%</span>
                </span>
            </div>""", unsafe_allow_html=True)
        with cb:
            if st.button("✕", key=f"rm_{sym}", help=f"Remove {sym}"):
                to_remove = sym

    if to_remove:
        st.session_state.watchlist.remove(to_remove)
        st.rerun()

    st.markdown("---")

    st.markdown("**Stock Ticker**")
    ticker = st.text_input("", value="TCS.NS",
                           help="TCS.NS, RELIANCE.NS, AAPL, GOOGL",
                           label_visibility="collapsed")

    st.markdown("**Select Models**")
    use_lr = st.checkbox("Linear Regression",       value=True)
    use_dt = st.checkbox("Decision Tree Regressor", value=True)
    use_rf = st.checkbox("Random Forest Regressor", value=True)

    run_btn = st.button("🚀 Run Prediction", use_container_width=True)

    st.markdown("---")
    st.markdown("""
**🇮🇳 Indian (NSE):** `TCS.NS` `RELIANCE.NS` `INFY.NS` `HDFCBANK.NS`

**🌍 US Stocks:** `AAPL` `MSFT` `GOOGL` `TSLA`
""")

st.markdown('<div class="main-title">📈 Stock Market Price Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

if not run_btn:
    st.info("👈 Enter a stock ticker in the sidebar and click **Run Prediction** to start.")

else:
    selected_models = []
    if use_lr: selected_models.append("Linear Regression")
    if use_dt: selected_models.append("Decision Tree")
    if use_rf: selected_models.append("Random Forest")

    if not selected_models:
        st.error("Please select at least one model.")
        st.stop()

    # Download
    end_date   = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=5)).strftime("%Y-%m-%d")

    with st.spinner(f"Downloading data for **{ticker}**..."):
        df = download_stock_data(ticker, start_date, end_date)

    if df is None or df.empty:
        st.error(f"Could not download data for **{ticker}**. Check the ticker symbol.")
        st.stop()

    with st.spinner("Calculating technical indicators..."):
        df = add_technical_indicators(df)

    lo52, hi52, cur = week52_data(ticker)

    if lo52 and hi52 and hi52 > lo52:
        pct_pos = min(max((cur - lo52) / (hi52 - lo52) * 100, 1), 99)

        st.markdown("#### 📊 52-Week Price Range")
        c1, c2, c3 = st.columns(3)
        c1.metric("52W Low",  f"₹{lo52:,.2f}")
        c2.metric("Current",  f"₹{cur:,.2f}")
        c3.metric("52W High", f"₹{hi52:,.2f}")

        st.markdown(f"""
        <div class="week52-wrap">
            <div class="week52-label">
                <span>₹{lo52:,.0f}</span>
                <span style="color:#94a3b8;font-size:0.7rem">
                    Currently at <b>{pct_pos:.0f}%</b> of 52-week range
                </span>
                <span>₹{hi52:,.0f}</span>
            </div>
            <div class="week52-bar">
                <div class="week52-dot" style="left:{pct_pos:.1f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 52-week sparkline chart
        df_52 = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if isinstance(df_52.columns, pd.MultiIndex):
            df_52.columns = df_52.columns.get_level_values(0)

        if not df_52.empty:
            close_52 = df_52["Close"].values.flatten()
            dates_52 = df_52.index
            line_col = "#22c55e" if close_52[-1] >= close_52[0] else "#ef4444"

            fig_52, ax_52 = plt.subplots(figsize=(12, 2.8))
            fig_52.patch.set_facecolor("#0f172a")
            ax_52.set_facecolor("#0f172a")

            ax_52.plot(dates_52, close_52, color=line_col, linewidth=1.5)
            ax_52.fill_between(dates_52, close_52, alpha=0.1, color=line_col)

            # Mark low / high
            idx_lo = close_52.argmin()
            idx_hi = close_52.argmax()
            ax_52.scatter(dates_52[idx_lo], close_52[idx_lo], color="#ef4444", zorder=5, s=45)
            ax_52.scatter(dates_52[idx_hi], close_52[idx_hi], color="#22c55e", zorder=5, s=45)
            ax_52.annotate(f"Low ₹{close_52[idx_lo]:,.0f}",
                           xy=(dates_52[idx_lo], close_52[idx_lo]),
                           xytext=(6, -16), textcoords="offset points",
                           color="#ef4444", fontsize=7.5)
            ax_52.annotate(f"High ₹{close_52[idx_hi]:,.0f}",
                           xy=(dates_52[idx_hi], close_52[idx_hi]),
                           xytext=(6, 6), textcoords="offset points",
                           color="#22c55e", fontsize=7.5)

            ax_52.tick_params(colors="#64748b", labelsize=7)
            for sp in ax_52.spines.values():
                sp.set_edgecolor("#334155")
            ax_52.grid(alpha=0.15, color="#334155")
            ax_52.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            ax_52.set_title(f"{ticker} — Last 52 Weeks", fontsize=10, color="#94a3b8", pad=5)
            plt.tight_layout()
            st.pyplot(fig_52)
            plt.close(fig_52)

    st.markdown("---")

    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    with st.spinner("Training models..."):
        models, predictions = train_all_models(X_train, X_test, y_train, selected_models)

    st.markdown("#### 🔮 Next Day Price Prediction")
    cols = st.columns(len(predictions))
    for i, (name, model) in enumerate(models.items()):
        next_price    = predict_next_day(model, X, scaler)
        rmse, mae, r2 = evaluate_model(y_test, predictions[name])
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{name}</h3>
                <p>₹ {next_price:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")

    last_close = df["Close"].iloc[-1]
    st.markdown(f"**Last Known Close:** ₹ {last_close:,.2f}  |  **Date:** {df.index[-1].strftime('%d %b %Y')}")

    st.markdown("#### 📈 Visual Analysis")
    tab1, tab2 = st.tabs(["Price & Moving Averages", "Actual vs Predicted"])
    test_dates = df.index[-len(y_test):]

    with tab1:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["Close"], label="Close Price", color="#1a1a2e", linewidth=1.5)
        if "SMA_20" in df.columns:
            ax.plot(df.index, df["SMA_20"], label="SMA 20", color="#667eea", linewidth=1, linestyle="--")
        if "SMA_50" in df.columns:
            ax.plot(df.index, df["SMA_50"], label="SMA 50", color="#f64f59", linewidth=1, linestyle="--")
        ax.set_title(f"{ticker} — Historical Price with Moving Averages", fontsize=13, fontweight="bold")
        ax.set_ylabel("Price (₹)")
        ax.legend(); ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(test_dates, y_test.values, label="Actual Price", color="#1a1a2e", linewidth=2)
        clrs = ["#667eea", "#f64f59", "#11998e"]
        for i, (name, pred) in enumerate(predictions.items()):
            ax.plot(test_dates, pred, label=f"{name} (Predicted)",
                    color=clrs[i % 3], linewidth=1.5, linestyle="--", alpha=0.85)
        ax.set_title(f"{ticker} — Actual vs Predicted Closing Price", fontsize=13, fontweight="bold")
        ax.set_ylabel("Price (₹)")
        ax.legend(); ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="footer">Stock Market Price Predictor — For educational use only</div>',
                unsafe_allow_html=True)
