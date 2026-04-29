import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("📈 Stock Beta, Dashboard & Prediction Tool")

# --- USER INPUT ---
stock_symbol = st.text_input("Enter the stock ticker (example: TSLA):", "TSLA")
market_symbol = st.text_input("Enter the market index ticker (example: ^GSPC):", "^GSPC")
period = st.selectbox("Select data period:", ("1mo", "3mo", "6mo", "1y", "2y", "5y"))

if st.button("Fetch & Analyze Data"):

    # --- DOWNLOAD DATA ---
    st.write(f"Fetching data for **{stock_symbol}** and **{market_symbol}**...")
    stock_data = yf.download(stock_symbol, period=period)
    market_data = yf.download(market_symbol, period=period)

    if stock_data.empty or market_data.empty:
        st.error("Error: Could not download data. Check ticker symbols.")
    else:
        # --- CALCULATE RETURNS ---
        stock_data["Return"] = stock_data["Close"].pct_change()
        market_data["Market_Return"] = market_data["Close"].pct_change()
        data = pd.concat([stock_data["Return"], market_data["Market_Return"]], axis=1).dropna()

        # --- CALCULATE BETA ---
        cov_matrix = np.cov(data["Return"], data["Market_Return"])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        st.metric(label=f"📌 Beta of {stock_symbol}", value=f"{beta:.2f}")

        # --- DASHBOARD CHARTS ---
        st.subheader("Cumulative Returns")
        data["Stock_Cum"] = (1 + data["Return"]).cumprod()
        data["Market_Cum"] = (1 + data["Market_Return"]).cumprod()

        fig1, ax1 = plt.subplots()
        ax1.plot(data["Stock_Cum"], label=stock_symbol)
        ax1.plot(data["Market_Cum"], label=market_symbol)
        ax1.set_ylabel("Cumulative Value")
        ax1.legend()
        st.pyplot(fig1)

        st.subheader("📈 Daily Returns Scatter Plot")
        fig2, ax2 = plt.subplots()
        ax2.scatter(data["Market_Return"], data["Return"], alpha=0.6)
        ax2.set_xlabel("Market Daily Return")
        ax2.set_ylabel(f"{stock_symbol} Daily Return")

        # Regression line
        lr = LinearRegression()
        lr.fit(data[["Market_Return"]], data["Return"])
        x_range = np.linspace(data["Market_Return"].min(), data["Market_Return"].max(), 100)
        y_pred = lr.predict(x_range.reshape(-1, 1))
        ax2.plot(x_range, y_pred, color="red", linestyle="--", label="Prediction Line")
        ax2.legend()

        st.pyplot(fig2)

        st.subheader("📉 Regression Prediction Summary")
        st.write(f"Regression slope (approx Beta): **{lr.coef_[0]:.2f}**")
        st.write(f"Regression intercept: **{lr.intercept_:.2f}**")