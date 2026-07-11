# ================================================================
#   StockIQ — Real LSTM Price Prediction Model
#   File Name  : lstm_model.py
#   Location   : StockProject/lstm_model.py
#
#   STEP 1 → Install libraries (run once in terminal):
#       pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
#
#   STEP 2 → Run this file:
#       python lstm_model.py
#
#   What it does:
#   - Downloads real stock data from Yahoo Finance
#   - Trains an LSTM neural network
#   - Predicts next 30 days of prices
#   - Saves a chart image: prediction_chart.png
# ================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # saves chart to file (no popup window needed)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Step 1: Download real stock data ──────────────────────────
print("\n🔽 Downloading stock data from Yahoo Finance...")

try:
    import yfinance as yf
    STOCK   = 'RELIANCE.NS'   # Change to any stock: TCS.NS, INFY.NS, AAPL, TSLA
    PERIOD  = '2y'             # 2 years of historical data
    df = yf.download(STOCK, period=PERIOD, auto_adjust=True)
    prices = df['Close'].values.flatten().astype(float)
    print(f"✅ Downloaded {len(prices)} days of {STOCK} data")
    print(f"   Price range: ₹{prices.min():.2f} → ₹{prices.max():.2f}")
except Exception as e:
    print(f"⚠️  Could not download data ({e})")
    print("   Using simulated data instead...")
    # Fallback: generate realistic fake price data
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(500) * 15) + 2600
    prices = np.maximum(prices, 1000).astype(float)
    STOCK = 'RELIANCE (Simulated)'

# ── Step 2: Prepare data for LSTM ─────────────────────────────
print("\n⚙️  Preparing data for LSTM...")

from sklearn.preprocessing import MinMaxScaler

# Scale prices between 0 and 1 (LSTM works better with scaled data)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(prices.reshape(-1, 1))

# Build sequences: use last 60 days to predict next day
LOOK_BACK = 60
X, y = [], []
for i in range(LOOK_BACK, len(scaled)):
    X.append(scaled[i - LOOK_BACK:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape for LSTM: [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train (80%) and test (20%)
split    = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# ── Step 3: Build the LSTM Model ──────────────────────────────
print("\n🧠 Building LSTM Neural Network...")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    USE_TENSORFLOW = True
except ImportError:
    print("⚠️  TensorFlow not installed. Run: pip install tensorflow")
    print("   Showing prediction with simple moving average instead...")
    USE_TENSORFLOW = False

if USE_TENSORFLOW:
    model = Sequential([
        # Layer 1: LSTM with 64 memory units, returns sequences for next layer
        LSTM(units=64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),  # Dropout prevents overfitting (randomly turns off 20% of neurons)

        # Layer 2: LSTM with 32 memory units
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),

        # Layer 3: Dense output (predicts 1 value)
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # ── Step 4: Train the Model ──────────────────────────────
    print("\n🚀 Training LSTM model (this takes 1–3 minutes)...")

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs          = 30,           # max 30 rounds of training
        batch_size      = 32,
        validation_split= 0.1,
        callbacks       = [early_stop],
        verbose         = 1
    )

    print(f"\n✅ Training complete! Stopped at epoch {len(history.history['loss'])}")

    # ── Step 5: Evaluate on Test Data ────────────────────────
    print("\n📊 Evaluating model on test data...")
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    pred_scaled = model.predict(X_test, verbose=0)
    predicted   = scaler.inverse_transform(pred_scaled).flatten()
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse      = mean_squared_error(actual, predicted)
    rmse     = np.sqrt(mse)
    mape     = mean_absolute_percentage_error(actual, predicted) * 100
    accuracy = 100 - mape

    print(f"   RMSE     : {rmse:.2f}  (lower = better)")
    print(f"   MAPE     : {mape:.2f}% (lower = better)")
    print(f"   Accuracy : {accuracy:.2f}%")

    # ── Step 6: Predict Next 30 Days ─────────────────────────
    print("\n🔮 Predicting next 30 days...")

    FORECAST_DAYS = 30
    last_60 = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
    future_prices = []

    for _ in range(FORECAST_DAYS):
        next_price = model.predict(last_60, verbose=0)[0, 0]
        future_prices.append(next_price)
        # Slide the window forward
        last_60 = np.append(last_60[:, 1:, :], [[[next_price]]], axis=1)

    future_prices = scaler.inverse_transform(
        np.array(future_prices).reshape(-1, 1)
    ).flatten()

    print(f"\n📅 30-Day Price Forecast for {STOCK}:")
    print(f"{'Day':<6} {'Predicted Price':>15}")
    print("-" * 22)
    for i, p in enumerate(future_prices, 1):
        print(f"  {i:<4} ₹{p:>12.2f}")

    current_price = prices[-1]
    final_price   = future_prices[-1]
    change_pct    = (final_price - current_price) / current_price * 100
    signal        = "🟢 BUY" if change_pct > 2 else ("🔴 SELL" if change_pct < -2 else "🟡 HOLD")

    print(f"\n{'='*40}")
    print(f"  Current Price : ₹{current_price:.2f}")
    print(f"  30-Day Target : ₹{final_price:.2f}")
    print(f"  Expected Move : {change_pct:+.2f}%")
    print(f"  AI Signal     : {signal}")
    print(f"{'='*40}\n")

    # ── Step 7: Save Chart ────────────────────────────────────
    print("📈 Saving prediction chart...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor('#030a0e')

    # ── Chart 1: Actual vs Predicted (Test Period) ──
    ax1 = axes[0]
    ax1.set_facecolor('#060f14')
    ax1.plot(actual,    color='#00c8ff', linewidth=2, label='Actual Price',    alpha=0.9)
    ax1.plot(predicted, color='#00ff88', linewidth=2, label='Predicted Price', alpha=0.9, linestyle='--')
    ax1.set_title(f'{STOCK} — LSTM Prediction vs Actual (Test Period)',
                  color='white', fontsize=13, pad=15)
    ax1.set_ylabel('Price (₹)', color='#4a7a8a')
    ax1.tick_params(colors='#4a7a8a')
    ax1.spines[:].set_color('#0d2d3a')
    ax1.legend(facecolor='#0a1a22', labelcolor='white', fontsize=10)
    ax1.grid(color='#0d2d3a', linewidth=0.5)
    ax1.text(0.02, 0.95, f'Accuracy: {accuracy:.1f}%  |  RMSE: {rmse:.2f}',
             transform=ax1.transAxes, color='#00ff88', fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='#0a1a22', alpha=0.8))

    # ── Chart 2: 30-Day Future Forecast ──
    ax2 = axes[1]
    ax2.set_facecolor('#060f14')

    # Show last 60 actual days + future 30 days
    history_prices = prices[-60:]
    x_hist   = np.arange(len(history_prices))
    x_future = np.arange(len(history_prices) - 1, len(history_prices) + len(future_prices))

    ax2.plot(x_hist, history_prices,
             color='#00c8ff', linewidth=2, label='Recent History', alpha=0.9)
    ax2.plot(x_future, np.concatenate([[history_prices[-1]], future_prices]),
             color='#c678dd', linewidth=2, linestyle='--', label='30-Day Forecast', alpha=0.9)

    # Confidence band
    std = np.std(np.diff(history_prices)) * np.sqrt(np.arange(1, len(future_prices) + 1))
    upper = np.concatenate([[history_prices[-1]], future_prices]) + np.concatenate([[0], std])
    lower = np.concatenate([[history_prices[-1]], future_prices]) - np.concatenate([[0], std])
    ax2.fill_between(x_future, upper, lower, color='#c678dd', alpha=0.1, label='Confidence Band')

    # Mark current and target
    ax2.axvline(x=len(history_prices)-1, color='#ffd700', linestyle=':', alpha=0.7, label='Today')

    ax2.set_title(f'{STOCK} — 30-Day Price Forecast  |  Signal: {signal}',
                  color='white', fontsize=13, pad=15)
    ax2.set_ylabel('Price (₹)', color='#4a7a8a')
    ax2.set_xlabel('Days', color='#4a7a8a')
    ax2.tick_params(colors='#4a7a8a')
    ax2.spines[:].set_color('#0d2d3a')
    ax2.legend(facecolor='#0a1a22', labelcolor='white', fontsize=10)
    ax2.grid(color='#0d2d3a', linewidth=0.5)

    plt.tight_layout()
    chart_path = 'prediction_chart.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#030a0e')
    plt.close()

    print(f"✅ Chart saved as: {chart_path}")
    print("   Open this file to see your prediction graph!\n")

    # ── Step 8: Save model ────────────────────────────────────
    model.save('lstm_model_saved.keras')
    print("✅ Model saved as: lstm_model_saved.keras")
    print("   You can load this later without re-training!\n")

else:
    # ── Fallback: Simple Moving Average Prediction ──────────
    print("\n📊 Using Simple Moving Average (no TensorFlow needed)...")

    window = 20
    ma20 = pd.Series(prices).rolling(window).mean().values
    ma50 = pd.Series(prices).rolling(50).mean().values

    # Simple linear extrapolation for 30 days
    slope = (prices[-1] - prices[-30]) / 30
    future = [prices[-1] + slope * i for i in range(1, 31)]

    print(f"\n30-Day SMA Forecast for {STOCK}:")
    for i, p in enumerate(future, 1):
        print(f"  Day {i:2d}: ₹{p:.2f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#030a0e')
    ax.set_facecolor('#060f14')
    ax.plot(prices[-120:], color='#00c8ff', linewidth=2, label='Price')
    x_fut = np.arange(120, 150)
    ax.plot(x_fut, future, color='#c678dd', linewidth=2, linestyle='--', label='Forecast')
    ax.set_title(f'{STOCK} — Simple Forecast (Install TensorFlow for LSTM)',
                 color='white')
    ax.tick_params(colors='#4a7a8a')
    ax.spines[:].set_color('#0d2d3a')
    ax.legend(facecolor='#0a1a22', labelcolor='white')
    ax.grid(color='#0d2d3a', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('prediction_chart.png', dpi=150, facecolor='#030a0e')
    plt.close()
    print("✅ Chart saved as prediction_chart.png")

print("\n🎉 Done! Your LSTM model run is complete.")
print("   Files created:")
print("   📈 prediction_chart.png  ← Open this to see the chart")
if USE_TENSORFLOW:
    print("   🧠 lstm_model_saved.keras ← Your trained model")
print()
