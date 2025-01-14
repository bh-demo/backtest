# GPT streamlit app based on my code to normalise balance sheet / fundamentals.
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import base64 # for image display.

# Actual code
def proc_vals2(tickers):
    tickers = tickers.split()
    data = []  # Store processed data for DataFrame

    for tck in tickers:
        tck_obj = yf.Ticker(tck)
        tinfo = tck_obj.info
        d = tck_obj.quarterly_balance_sheet

        # Extract equity
        try:
            if "Common Stock Equity" in d.index:
                equity = d.loc["Common Stock Equity"].iloc[0]
            elif "Stockholders Equity" in d.index:
                equity = d.loc["Stockholders Equity"].iloc[0]
            else:
                equity = None
        except (KeyError, IndexError):
            equity = None
        # Extract shares
        shares1 = tinfo.get("impliedSharesOutstanding", -1)
        shares2 = tinfo.get("sharesOutstanding", -1)
        try:
            shares3 = d.loc["Share Issued"].iloc[0]
        except (KeyError, IndexError):
            shares3 = -1
        shares = max(filter(lambda x: x not in [None, -1], [shares1, shares2, shares3]), default=None)

        # Current price in pounds
        cp = tinfo.get("currentPrice", 0) / 100.0 if tinfo.get("currentPrice") else None

        # Price-to-Earnings ratio
        forward_eps = tinfo.get("forwardEps", None)
        pe = cp / forward_eps if cp and forward_eps else tinfo.get("forwardPE", -1)

        # Dividend yield
        yld = tinfo.get("dividendYield", 0.0) * 100

        # Target price ratio
        tgt = tinfo.get("targetMedianPrice", 0) / cp if cp else None
        if tgt is not None and tgt > 5.0:
            tgt /= 100

        # Exchange rate adjustments
        currency_pair = tinfo.get("currency", "USD") + tinfo.get("financialCurrency", "USD") + "=X"
        exchange_info = yf.Ticker(currency_pair).info
        sf = (exchange_info.get("bid", 1) + exchange_info.get("ask", 1)) / 2

        if sf == 1.0:
            pe *= 100
        if tinfo.get("currency", "USD") in ["EUR", "USD"]:
            cp = cp * 100 if cp else None

        # Equity per share adjusted for exchange rate
        eqs = equity / (shares * sf) if equity and shares and sf and sf > 0 else None

        # Debt-to-Equity ratio
        try:
            long_term_debt = d.loc["Long Term Debt And Capital Lease Obligation"].iloc[0]
        except (KeyError, IndexError):
            long_term_debt = d.loc['Long Term Provisions'].iloc[0]
        try:
            current_debt = d.loc["Current Debt And Capital Lease Obligation"].iloc[0]
        except (KeyError, IndexError):
            try:
                current_debt = d.loc['Current Provisions'].iloc[0]
            except:
                current_debt = 0
        debt = (long_term_debt or 0) + (current_debt or 0)
        de = debt / equity if equity else None

        data.append({
            "Symbol": tinfo.get("symbol", "N/A"),
            "Equity/Share (Price Adjusted)": eqs / cp if eqs and cp else None,
            "Debt/Equity": de,
            "P/E": pe if pe != -1 else None,
            "Yield (%)": yld,
            "Target/Price": tgt,
        })

    # Replace None with NaN for compatibility with formatting
    df = pd.DataFrame(data).replace({None: np.nan})
    return df

def add_logo():
    # Display clickable logo at the top of the app
    st.markdown(
        """
        <div style="text-align: center;">
            <a href="https://www.bghtech.co.uk/" target="_blank">
                <img src="data:image/png;base64,{}" width="150">
            </a>
        </div>
        """.format(
        base64.b64encode(open("/backtest/Bghtech_logo.png", "rb").read()).decode()
    ),
        unsafe_allow_html=True
    )
    
# Streamlit app
def main():
    add_logo()
    st.title("Stocks - Fundamentals Analysis")
    st.write("Enter space-separated Yahoo stock symbols to analyze.")

    # Input field for stock symbols
    tickers = st.text_input("Stock Symbols", "AAPL MSFT TSLA NVDA META AMZN GOOG RMV.L PSN.L CNA.L BATS.L")

    if st.button("Analyze"):
        if tickers:
            try:
                # Process tickers and display DataFrame
                result_df = proc_vals2(tickers)
                st.dataframe(result_df.style.format(
                    {
                        "Equity/Share (Price Adjusted)": "{:.2f}",
                        "Debt/Equity": "{:.2f}",
                        "P/E": "{:.1f}",
                        "Yield (%)": "{:.1f}",
                        "Target/Price": "{:.2f}",
                    }
                ), use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter at least one stock symbol.")

if __name__ == "__main__":
    main()
