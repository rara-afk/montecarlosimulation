import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

st.title("Monte Carlo Stock Prediction Dashboard")

# USER INPUT

ticker = st.text_input("Stock Symbol", "AAPL")
days = st.slider("Prediction Days", 10, 180, 30)
simulations = st.slider("Number of Simulations", 10 ,100, 5000, 1000)

# DOWNLOAD DATA

data = yf.download(ticker, start="2025-01-01")
data.columns = data.columns.get_level_values(0)

if data.empty:
    st.error("Stock data not found")
    st.stop()

close = data["Close"]

# HISTORICAL CHART

st.subheader("Historical Stock Price")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=close,
        mode="lines",
        name="Stock Price"
    )
)

fig.update_layout(
    title=f"{ticker} Historical Price",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# RETURNS

returns = close.pct_change().dropna()

mu = returns.mean()
sigma = returns.std()

S0 = float(close.iloc[-1])

# MONTE CARLO SIMULATION

results = np.zeros((days, simulations))

for i in range(simulations):

    price = S0

    for t in range(days):

        price = price * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal())

        results[t, i] = price


# FUTURE PRICE CHART

fig2 = go.Figure()

for i in range(simulations):
    fig2.add_trace(
        go.Scatter(
            y=results[:, i],
            mode="lines",
            line=dict(width=1),
            showlegend=False
        )
    )

fig2.update_layout(
    title="Monte Carlo Simulation - Future Prices",
    xaxis_title="Days",
    yaxis_title="Price"
)

st.plotly_chart(fig2)

# FINAL PRICE ANALYSIS

final_prices = np.array(results[-1])

expected_price = np.mean(final_prices)

expected_return = (expected_price - S0) / S0

min_price = np.min(final_prices)

max_price = np.max(final_prices)

# RISK METRICS

sim_returns = (final_prices - S0) / S0

VaR_95 = np.percentile(sim_returns, 5)

ES_95 = sim_returns[sim_returns <= VaR_95].mean()

prob_profit = np.sum(final_prices > S0) / simulations

risk_free_rate = 0.02 / 252

sharpe = (mu - risk_free_rate) / sigma

# DASHBOARD METRICS

st.subheader("Investment Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Current Price", round(S0,2))
col2.metric("Expected Price", round(expected_price,2))
col3.metric("Expected Return", str(round(expected_return*100,2))+" %")

col4, col5, col6 = st.columns(3)

col4.metric("Value at Risk (95%)", str(round(VaR_95*100,2))+" %")
col5.metric("Expected Shortfall", str(round(ES_95*100,2))+" %")
col6.metric("Probability of Profit", str(round(prob_profit*100,2))+" %")

st.metric("Sharpe Ratio", round(sharpe,2))