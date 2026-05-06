import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- UI Configuration ---
st.set_page_config(page_title="TradingView AI Pro", layout="wide")
st.title("📊 TradingView-Style AI Terminal")

# --- Sidebar Management ---
st.sidebar.header("Watchlist & Settings")
stock_dict = {
    "Nifty 50": "^NSEI",
    "Bank Nifty": "^NSEBANK",
    "Sensex": "^BSESN",
    "Reliance": "RELIANCE.NS",
    "Jio Finance": "JIOFIN.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Power": "TATAPOWER.NS",
    "Suzlon": "SUZLON.NS"
}
selected_name = st.sidebar.selectbox("Select Asset", list(stock_dict.keys()))
ticker = stock_dict[selected_name]

chart_style = st.sidebar.selectbox("Chart Style", ["Candlestick", "Hollow", "Line"])
time_period = st.sidebar.selectbox("Timeframe", ["1y", "2y", "5y"])

# --- Data Engine ---
@st.cache_data
def load_data(ticker, period):
    df = yf.download(ticker, period=period, interval="1d")
    # Flatten columns if multi-index (common in newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = load_data(ticker, time_period)
ticker_obj = yf.Ticker(ticker)

# --- Technical Indicators Calculation ---
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# RSI Calculation
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# --- TradingView-Style Multi-Chart ---
st.subheader(f"Technical Analysis: {selected_name}")

# Create subplots: Price (row 1), RSI (row 2)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, subplot_titles=(f'{selected_name} Price & Volume', 'RSI (14)'),
                    row_heights=[0.7, 0.3])

# 1. Main Price Chart
if chart_style == "Candlestick":
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
elif chart_style == "Hollow":
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], 
                                 increasing_line_color='cyan', decreasing_line_color='red', increasing_fillcolor='rgba(0,0,0,0)', name="Price"), row=1, col=1)
else:
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', line=dict(color='#00ffcc'), name="Price"), row=1, col=1)

# Add Moving Averages
fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], line=dict(color='orange', width=1), name="MA20"), row=1, col=1)
fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], line=dict(color='magenta', width=1), name="MA50"), row=1, col=1)

# 2. RSI Chart
fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], line=dict(color='#f2f2f2', width=1), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig.update_layout(template="plotly_dark", height=700, showlegend=False, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- Pro Bottom Panel: News, Sentiment, and AI ---
st.divider()
col_news, col_ai = st.columns([1, 1])

with col_news:
    st.subheader("📰 Market Intelligence")
    news_list = ticker_obj.news
    if news_list:
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        for item in news_list[:5]:
            title = item.get('title', 'No Title Available')  # FIXED: Handles missing titles
            publisher = item.get('publisher', 'Unknown Source')
            st.markdown(f"**{title}**  \n*{publisher}*")
            
            # Sentiment Math
            vs = analyzer.polarity_scores(title)
            sentiments.append(vs['compound'])
        
        # Sentiment Gauage
        avg_s = np.mean(sentiments) if sentiments else 0
        if avg_s > 0.05:
            st.success(f"BULLISH SENTIMENT: {avg_s:.2f}")
        elif avg_s < -0.05:
            st.error(f"BEARISH SENTIMENT: {avg_s:.2f}")
        else:
            st.warning(f"NEUTRAL SENTIMENT: {avg_s:.2f}")
    else:
        st.info("No recent news found for this asset.")

with col_ai:
    st.subheader("🤖 AI Forecast Model")
    if st.button("Generate Prediction"):
        with st.spinner("Processing Time-Series Data..."):
            prices = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(prices)
            
            # Sequence Prep
            X = []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i, 0])
            X = np.array(X)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Quick Model
            model = Sequential([LSTM(50, input_shape=(60, 1)), Dense(1)])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, scaled_data[60:], epochs=1, verbose=0)
            
            # Predict Next Day
            last_batch = scaled_data[-60:].reshape(1, 60, 1)
            prediction = scaler.inverse_transform(model.predict(last_batch))
            
            st.metric("LSTM Predicted Next Close", f"₹{prediction[0][0]:.2f}")
            st.caption("Disclaimer: AI predictions are for educational use and not financial advice.")