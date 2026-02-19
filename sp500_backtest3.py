# GPT streamlit app based on my code to normalise balance sheet / fundamentals.
# Growth estimate added

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import base64  # for image display.

# Actual code
def proc_vals2(tickers):
    tickers = tickers.split()
    data = []  # Store processed data for DataFrame

    for tck in tickers:
        tck_obj = yf.Ticker(tck)
        tinfo = tck_obj.info
        d = tck_obj.quarterly_balance_sheet
        e = tck_obj.earnings_estimate
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
        forward_eps = e['yearAgoEps'].iloc[-1]
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

        if tinfo.get("currency", "GBP") in ["GBP","EUR", "USD"]:
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

        # Growth Estimate (Annualized Earnings Growth)
        growth_est0 = e['growth'].iloc[-2]  # growth curr year
        growth_est1 = e['growth'].iloc[-1]  # growth next year
        if growth_est0 is not None:
            growth_est0 *= 100  # Convert to percentage
        if growth_est1 is not None:
            growth_est1 *= 100  # Convert to percentage

        data.append({
            "Symb": tinfo.get("symbol", "N/A"),
            "Eqty/Shr": eqs / cp if eqs and cp else None,
            "Dbt/Eqty": de,
            "Net Eqty": eqs/cp - ((eqs/cp)*de) if eqs and cp else None,
            "P/E": pe if pe != -1 else None,
            "Yld (%)": yld,
            "Tgt/Price": tgt,
            "G_0y(%)": growth_est0, # Format name must be same as in df
            "G_1y(%)": growth_est1,
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
        base64.b64encode(open("Bghtech_logo.PNG", "rb").read()).decode()
    ),
        unsafe_allow_html=True
    )

# choice dict
choice_dict = {
    "manual symbols": "AAPL MSFT PSN.L ",
    "airlines": 'IAG.L EZJ.L WIZZ.L AF.PA LHA.DE RYA.IR AAL DAL JBLU',
    "banks": 'HSBA.L BARC.L LLOY.L NWG.L MTRO.L JPM BAC GS WFC ',
    "builders":'RMV.L PSN.L VTY.L BKG.L BTRW.L TW.L BWY.L CRST.L TOL PHM DHI LEN',
    "BH_select": 'VOD.L CURY.L BARC.L LAND.L CRST.L NWG.L HSBA.L BATS.L SQZ.L TW.L S32.L PSN.L SHEL.L MNG.L VTY.L FRES.L o PRU.L AV.L TSCO.L SSE.L CNA.L PHNX.L RIO.L PNN.L GFRD.L IMB.L LGEN.L IAG.L KGF.L DOCS.L EZJ.L MTRO.L UTG.L HLN.L CTEC.L BTRW.L',
    "BH_":'CHG.L DSCV.L MRO.L SNR.L SPX.L WEIR.L ATYM.L HOC.L TCAP.L VCT.L TBCG.L DDOG VICI FSLR HPE KHC IVZ OMC TER AMCR FDS CEG',
    "Real Estate ETFs": 'PLD EQIX WELL SPG O DLR PSA VICI EXR AVB',
    "Reits": 'BLND.L LAND.L UTG.L O ADC SPG NNN FRT KIM WPC',
    "Water": 'PNN.L UU.L SVT.L AWK WTRG AWR CWT ARIS SJW',
    "Insurers": 'AV.L LGEN.L MNG.L PHNX.L PRU.L',
    "US_stocks": 'NVDA MSFT AAPL GOOG META TSLA AMZN VZ CVX INTC AMD IBM CSCO KO PFE ACN V PYPL GE CRM UNH'
    }

# Streamlit app
def main():
    add_logo()
    st.title("Stocks - Fundamentals Analysis")
    st.write("Enter space-separated Yahoo stock symbols to analyze.")

    # Dropdown menu for stock selection
    stock_choice = st.selectbox(
        "Select Stocks:", 
        options=["manual symbols","ISA", "SIPP", "airlines", "banks", "builders","BH_select", "couriers", "high div", "Real Estate ETFs", "Reits","US_stocks"]
    )

    # Input field for stock symbols
    tickers = st.text_input("Stock Symbols", choice_dict[stock_choice])

    if st.button("Analyse"):
        if tickers:
            try:
                # Process tickers and display DataFrame
                result_df = proc_vals2(tickers)
                st.dataframe(result_df.style.format(
                    {
                        "Eqty/Shr/price": "{:.2f}",
                        "Dbt/Eqty": "{:.2f}",
                        "P/E": "{:.1f}",
                        "Yld (%)": "{:.1f}",
                        "Tgt/Price": "{:.2f}",
                        "G_0y(%)": "{:.1f}",  # Format name must be same as in df
                        "G_1y(%)": "{:.1f}",
                    }
                ), use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter at least one stock symbol.")

if __name__ == "__main__":
    main()








