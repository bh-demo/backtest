"""
Streamlit Discounted Cash Flow (DCF) Fair Price Valuation App — with Inflation, Dividends & Currency Normalization

Features added/updated:
- Fetches dividends history from Yahoo and estimates an average annual dividend per share.
- Adds an Inflation Rate input (default 2.5%).
- Computes two DCF valuations per ticker:
  1. Base DCF (using nominal discount rate) — without dividends.
  2. Inflation-adjusted DCF (using a real discount rate derived from nominal discount rate and inflation).
- For both of the above, also compute an "+ Dividends" variant where the present value of projected dividends per share is added to the per-share DCF fair price.
- Handles .L (London) tickers and GBX/GBp quoted prices by normalizing pence→pounds.
- **Normalizes balance sheet currencies**: if balance sheet financials are reported in a different currency than the share price, attempts to convert balance-sheet cash/debt to the share-price currency using Yahoo FX tickers (e.g. EURGBP=X).

How to run:
1. pip install streamlit yfinance pandas numpy matplotlib
2. streamlit run streamlit_dcf_app.py

Notes & reasoning about currency handling:
- Yahoo sometimes quotes London-listed share prices in **pence (GBX)** while the "currency" field may say 'GBX', 'GBp' or 'GBP'. The app treats any GB* indication as pence and divides prices by 100.
- Companies can report balance sheet items in a different financial currency (Vodafone plc VOD.L historically reported some figures in EUR while the share trades in GBP). We detect `financialCurrency` or `currency` from yfinance info. If the balance-sheet currency differs from the price currency, the app attempts to fetch the FX spot rate using Yahoo FX symbols (e.g. 'EURGBP=X') and convert balance-sheet cash/debt into the price currency before using them in valuation.
- If FX lookup fails, the app warns and proceeds without conversion (i.e., uses raw numbers). This is safer than silently producing wrong valuations.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict

st.set_page_config(page_title="DCF Fair Price Valuation (Currency-normalized)", layout="wide")

# ------------------------- Helper functions -------------------------
@st.cache_data(show_spinner=False)
def fetch_ticker_data(ticker: str) -> Dict:
    """Fetch serializable pieces from yfinance Ticker. Return dicts/dataframes/series only.
    Includes dividends series (per-share dividend history).
    """
    t = yf.Ticker(ticker)
    try:
        info = dict(t.info) if hasattr(t, 'info') else {}
    except Exception:
        info = {}

    def safe_df(df):
        if df is None:
            return pd.DataFrame()
        return df.copy()

    fin = safe_df(getattr(t, "financials", pd.DataFrame()))
    cf = safe_df(getattr(t, "cashflow", pd.DataFrame()))
    bs = safe_df(getattr(t, "balance_sheet", pd.DataFrame()))

    # dividends is a pandas Series indexed by date, values are dividend per share
    try:
        div = t.dividends.copy() if getattr(t, 'dividends', None) is not None else pd.Series(dtype=float)
    except Exception:
        div = pd.Series(dtype=float)

    return {"info": info, "financials": fin, "cashflow": cf, "balance_sheet": bs, "dividends": div}


def detect_price_currency_and_scale(info: dict, ticker: str) -> Dict[str, Optional[object]]:
    """Detect the currency that the share price is quoted in and whether we must scale it (e.g. GBX->divide by 100).
    Returns a dict with keys: 'price_currency' (e.g. 'GBP'), 'price_scale' (multiplier to apply to raw price), 'notes'
    """
    notes = []
    # common fields from yfinance
    currency = info.get('currency') or info.get('quoteCurrency') or None
    financial_currency = info.get('financialCurrency') if 'financialCurrency' in info else None

    price_currency = None
    price_scale = 1.0

    if currency:
        cur = str(currency).upper()
        # treat GBX/GBp/GBP distinctions
        if cur in ('GBX', 'GBP', 'GBP '):
            # GBX often means pence; heuristic: if ticker ends with .L assume pence
            if cur == 'GBX' or cur.startswith('GBP') and ticker.endswith('.L'):
                price_currency = 'GBP'
                price_scale = 1.0 / 100.0  # convert pence to pounds
                notes.append('Detected GBX/GBp - scaling price by 1/100 to get GBP')
            else:
                price_currency = 'GBP'
        else:
            # generic assignment
            price_currency = cur
    else:
        # fallback: if ticker ends with .L assume GBP/pence
        if ticker.endswith('.L'):
            price_currency = 'GBP'
            price_scale = 1.0 / 100.0
            notes.append('No currency field; ticker ends with .L — assume pence and scale by 1/100')

    return {'price_currency': price_currency, 'price_scale': price_scale, 'notes': notes, 'financial_currency': financial_currency}


def fx_rate(from_cur: str, to_cur: str) -> Optional[float]:
    """Attempt to fetch FX spot rate from Yahoo using ticker like EURGBP=X. Returns rate as float or None.
    If from_cur == to_cur returns 1.0.
    """
    if from_cur is None or to_cur is None:
        return None
    from_cur = str(from_cur).upper()
    to_cur = str(to_cur).upper()
    if from_cur == to_cur:
        return 1.0
    try:
        pair = f"{from_cur}{to_cur}=X"
        t = yf.Ticker(pair)
        hist = t.history(period='1d')
        if hist is not None and not hist.empty:
            rate = hist['Close'].iloc[-1]
            return float(rate)
        # try inverted pair
        inv = f"{to_cur}{from_cur}=X"
        t2 = yf.Ticker(inv)
        hist2 = t2.history(period='1d')
        if hist2 is not None and not hist2.empty:
            rate_inv = hist2['Close'].iloc[-1]
            return 1.0 / float(rate_inv)
    except Exception:
        return None
    return None


def normalize_balance_sheet_currency(cash: Optional[float], debt: Optional[float], bs_currency: Optional[str], price_currency: Optional[str], ticker: str) -> Dict[str, Optional[float]]:
    """Convert cash and debt values from bs_currency to price_currency using FX rates if needed.
    Returns possibly converted {'cash':..., 'debt':..., 'fx_rate': ...} and warns via Streamlit if conversion fails.
    """
    if cash is None and debt is None:
        return {'cash': cash, 'debt': debt, 'fx_rate': None}

    if bs_currency is None or price_currency is None:
        return {'cash': cash, 'debt': debt, 'fx_rate': None}

    bs_cur = str(bs_currency).upper()
    pr_cur = str(price_currency).upper()

    # Normalize some common variants
    if bs_cur in ('GBX', 'GBP', 'GBP '):
        bs_cur = 'GBP'
    if pr_cur in ('GBX', 'GBP', 'GBP '):
        pr_cur = 'GBP'

    if bs_cur == pr_cur:
        return {'cash': cash, 'debt': debt, 'fx_rate': 1.0}

    rate = fx_rate(bs_cur, pr_cur)
    if rate is None:
        st.warning(f"Could not fetch FX rate to convert balance-sheet currency {bs_cur} → {pr_cur} for {ticker}. Using raw numbers.")
        return {'cash': cash, 'debt': debt, 'fx_rate': None}

    converted_cash = float(cash) * rate if cash is not None else None
    converted_debt = float(debt) * rate if debt is not None else None
    return {'cash': converted_cash, 'debt': converted_debt, 'fx_rate': rate}


def extract_historical_fcf(cashflow: pd.DataFrame, financials: pd.DataFrame) -> pd.Series:
    if cashflow is None or cashflow.empty:
        return pd.Series(dtype=float)

    op_candidates = [r for r in cashflow.index if 'operat' in r.lower()]
    capex_candidates = [r for r in cashflow.index if 'capital' in r.lower() or 'capex' in r.lower()]

    if not op_candidates or not capex_candidates:
        return pd.Series(dtype=float)

    op_row = cashflow.loc[op_candidates[0]]
    capex_row = cashflow.loc[capex_candidates[0]]

    fcf = op_row.apply(pd.to_numeric, errors='coerce') - capex_row.apply(pd.to_numeric, errors='coerce')
    fcf = fcf.dropna()
    fcf.index = pd.to_datetime(fcf.index)
    fcf = fcf.sort_index()
    fcf.index = fcf.index.year.astype(str)
    return fcf


def get_balance_sheet_items(bs: pd.DataFrame, info: dict) -> Dict[str, Optional[float]]:
    cash = None
    total_debt = None

    if bs is not None and not bs.empty:
        cash_candidates = [r for r in bs.index if 'cash' in r.lower()]
        debt_candidates = [r for r in bs.index if 'debt' in r.lower()]

        if cash_candidates:
            try:
                cash = float(bs.loc[cash_candidates[0]].dropna().iloc[0])
            except Exception:
                cash = None
        if debt_candidates:
            try:
                total_debt = float(bs.loc[debt_candidates[0]].dropna().iloc[0])
            except Exception:
                total_debt = None

    if cash is None:
        cash = info.get('totalCash') or info.get('cash')
    if total_debt is None:
        total_debt = info.get('totalDebt') or info.get('longTermDebt')

    try:
        cash = float(cash) if cash is not None else None
    except Exception:
        cash = None
    try:
        total_debt = float(total_debt) if total_debt is not None else None
    except Exception:
        total_debt = None

    return {"cash": cash, "debt": total_debt}


def dcf_valuation(latest_fcf: float, years: int, growth_rates: list, discount_rate: float, terminal_growth: float,
                  cash: Optional[float], debt: Optional[float], shares_outstanding: Optional[float]) -> Dict:
    """Project FCF, discount to present, add terminal value, compute per-share fair value.
    Returns enterprise/equity values and per-share fair price.
    """
    if len(growth_rates) == 0:
        growth_rates = [0.05] * years
    if len(growth_rates) < years:
        growth_rates = growth_rates + [growth_rates[-1]] * (years - len(growth_rates))
    growth_rates = growth_rates[:years]

    fcf_forecast = []
    fcf = latest_fcf
    for g in growth_rates:
        fcf = fcf * (1 + g)
        fcf_forecast.append(fcf)

    discounted = []
    for t, f in enumerate(fcf_forecast, start=1):
        discounted.append(f / ((1 + discount_rate) ** t))

    last_fcf = fcf_forecast[-1]
    if discount_rate <= terminal_growth:
        terminal_value = np.nan
    else:
        terminal_value = last_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)

    discounted_terminal = terminal_value / ((1 + discount_rate) ** years) if not np.isnan(terminal_value) else np.nan

    enterprise_value = np.nansum(discounted) + discounted_terminal

    adj = 0.0
    if cash is not None:
        adj += cash
    if debt is not None:
        adj -= debt

    equity_value = enterprise_value + adj

    fair_price = None
    if shares_outstanding and shares_outstanding > 0:
        fair_price = equity_value / shares_outstanding

    return {
        'fcf_forecast': fcf_forecast,
        'discounted': discounted,
        'terminal_value': terminal_value,
        'discounted_terminal': discounted_terminal,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'fair_price': fair_price
    }


def pv_of_projected_dividends(avg_dividend_per_share: float, growth_rates: list, discount_rate: float) -> float:
    """Project a stream of dividends per share using growth_rates and compute present value per share.
    If avg_dividend_per_share is None or 0, returns 0.
    """
    if avg_dividend_per_share is None or avg_dividend_per_share == 0:
        return 0.0
    pv = 0.0
    div = avg_dividend_per_share
    for t, g in enumerate(growth_rates, start=1):
        div = div * (1 + g)
        pv += div / ((1 + discount_rate) ** t)
    return pv


# ------------------------- Streamlit UI -------------------------
st.title("Discounted Cash Flow (DCF) — Fair Price Valuation (Inflation, Dividends & Currency)")
st.markdown("Enter one or more Yahoo Finance tickers (comma-separated) to compute DCF fair prices. The app will try to normalise currencies (price vs balance sheet) and convert pence→pounds for London tickers when necessary.")

with st.sidebar:
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, NVDA, META, O, VOD.L, HSBA.L, PSN.L, PHNX.L, AV.L, BARC.L, GFRD.L, PNN.L, SSE.L, TW.L, LGEN.L ")
    proj_years = st.number_input("Projection years", min_value=3, max_value=20, value=5, step=1)
    discount_rate = st.number_input("Discount rate (nominal, as decimal)", min_value=0.0, max_value=1.0, value=0.10, step=0.005, format="%.3f")
    inflation_rate = st.number_input("Inflation rate (as decimal)", min_value=0.0, max_value=0.20, value=0.025, step=0.001, format="%.3f")
    terminal_growth = st.number_input("Terminal growth rate (as decimal)", min_value=-0.05, max_value=0.10, value=0.02, step=0.001, format="%.3f")
    growth_mode = st.selectbox("Growth input mode", ["Single constant growth", "Year-by-year list", "Declining staging (High->Medium->Low)"])

    if growth_mode == "Single constant growth":
        growth_input = st.number_input("FCF growth rate (as decimal)", min_value=-0.5, max_value=1.0, value=0.10, step=0.01, format="%.3f")
        growth_list = [growth_input]
    elif growth_mode == "Year-by-year list":
        raw = st.text_area("Enter growth rates for each year, comma-separated (e.g. 0.2,0.15,0.1,0.08)")
        try:
            growth_list = [float(x.strip()) for x in raw.split(',') if x.strip()!='']
        except Exception:
            growth_list = [0.10]
    else:
        high = st.number_input("High growth rate (years 1-2)", value=0.20, format="%.3f")
        medium = st.number_input("Medium growth rate (years 3-5)", value=0.10, format="%.3f")
        low = st.number_input("Low growth rate (years 6+)", value=0.03, format="%.3f")
        growth_list = []
        for y in range(1, proj_years + 1):
            if y <= 2:
                growth_list.append(high)
            elif y <= 5:
                growth_list.append(medium)
            else:
                growth_list.append(low)

    run_button = st.button("Run valuation")


if not tickers_input.strip():
    st.info("Please enter at least one ticker symbol in the sidebar.")
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

results = []

if run_button:
    real_discount_rate = (1 + discount_rate) / (1 + inflation_rate) - 1 if inflation_rate is not None else discount_rate

    for tk in tickers:
        with st.spinner(f"Fetching data for {tk}..."):
            data = fetch_ticker_data(tk)

        info = data.get('info', {}) or {}
        cf = data.get('cashflow')
        fin = data.get('financials')
        bs = data.get('balance_sheet')
        dividends = data.get('dividends') if data.get('dividends') is not None else pd.Series(dtype=float)

        # detect price currency and scale factor (pence -> pounds)
        curinfo = detect_price_currency_and_scale(info, tk)
        price_currency = curinfo.get('price_currency')
        price_scale = curinfo.get('price_scale') or 1.0

        hist_fcf = extract_historical_fcf(cf, fin)

        latest_fcf = None
        if len(hist_fcf) > 0:
            latest_fcf = float(hist_fcf.dropna().iloc[-1])
        else:
            latest_fcf = info.get('freeCashflow') or info.get('freeCashFlow')

        if latest_fcf is None:
            st.error(f"Couldn't determine a usable latest free cash flow for {tk}. Skipping.")
            continue

        bal_items = get_balance_sheet_items(bs, info)
        cash_raw = bal_items.get('cash')
        debt_raw = bal_items.get('debt')

        # determine balance-sheet currency (financialCurrency) and convert to price currency if needed
        bs_currency = info.get('financialCurrency') or info.get('currency') or None
        normalized_bs = normalize_balance_sheet_currency(cash=cash_raw, debt=debt_raw, bs_currency=bs_currency, price_currency=price_currency, ticker=tk)
        cash = normalized_bs.get('cash')
        debt = normalized_bs.get('debt')
        fx_rate_used = normalized_bs.get('fx_rate')

        # estimate shares using market cap and normalized price
        shares = info.get('sharesOutstanding') or info.get('floatShares') or None
        if shares is None:
            price_for_shares = info.get('currentPrice') or info.get('previousClose')
            if price_for_shares is not None:
                price_for_shares = float(price_for_shares) * price_scale
            mcap = info.get('marketCap')
            if price_for_shares and mcap:
                try:
                    shares = float(mcap) / float(price_for_shares)
                except Exception:
                    shares = None

        # compute base DCF (nominal discount rate) using normalized cash/debt
        try:
            dcf_base = dcf_valuation(latest_fcf=latest_fcf, years=proj_years, growth_rates=growth_list,
                                     discount_rate=discount_rate, terminal_growth=terminal_growth,
                                     cash=cash, debt=debt, shares_outstanding=shares)
        except Exception as e:
            st.error(f"Error computing base DCF for {tk}: {e}")
            continue

        # compute inflation-adjusted DCF using real discount rate
        try:
            dcf_real = dcf_valuation(latest_fcf=latest_fcf, years=proj_years, growth_rates=growth_list,
                                     discount_rate=real_discount_rate, terminal_growth=terminal_growth,
                                     cash=cash, debt=debt, shares_outstanding=shares)
        except Exception as e:
            st.error(f"Error computing inflation-adjusted DCF for {tk}: {e}")
            continue

        # estimate average annual dividend per share from dividend history (last 3 years)
        avg_div_per_share = None
        try:
            if isinstance(dividends, (pd.Series, pd.DataFrame)) and len(dividends) > 0:
                div_series = dividends.copy()
                if isinstance(div_series, pd.DataFrame):
                    div_series = div_series.iloc[:, 0]
                div_series.index = pd.to_datetime(div_series.index)
                div_annual = div_series.resample('Y').sum()
                if len(div_annual) > 0:
                    avg_div_per_share = float(div_annual.tail(3).mean())
        except Exception:
            avg_div_per_share = None

        # compute PV of projected dividends per share (nominal and real)
        pv_div_nominal = pv_of_projected_dividends(avg_dividend_per_share=avg_div_per_share, growth_rates=growth_list, discount_rate=discount_rate)
        pv_div_real = pv_of_projected_dividends(avg_dividend_per_share=avg_div_per_share, growth_rates=growth_list, discount_rate=real_discount_rate)

        # per-share fair prices (base and with dividends)
        fair_base = dcf_base.get('fair_price')
        fair_base_plus_div = None
        if fair_base is not None:
            fair_base_plus_div = fair_base + pv_div_nominal

        fair_real = dcf_real.get('fair_price')
        fair_real_plus_div = None
        if fair_real is not None:
            fair_real_plus_div = fair_real + pv_div_real

        # current price handling (scale pence->pound if needed)
        current_price_raw = info.get('currentPrice') or info.get('previousClose')
        current_price = None
        if current_price_raw is not None:
            try:
                current_price = float(current_price_raw) * price_scale
            except Exception:
                current_price = None

        results.append({
            'ticker': tk,
            'latest_price': current_price,
            'price_currency': price_currency,
            'price_scale': price_scale,
            'market_cap': info.get('marketCap'),
            'latest_fcf': latest_fcf,
            'cash_raw': cash_raw,
            'debt_raw': debt_raw,
            'cash': cash,
            'debt': debt,
            'fx_rate_used': fx_rate_used,
            'shares': shares,
            'div_per_share': avg_div_per_share,
            'pv_div_nominal': pv_div_nominal,
            'pv_div_real': pv_div_real,
            'dcf_base': dcf_base,
            'dcf_real': dcf_real,
            'fair_base': fair_base,
            'fair_base_plus_div': fair_base_plus_div,
            'fair_real': fair_real,
            'fair_real_plus_div': fair_real_plus_div,
            'historical_fcf': hist_fcf
        })

    # build summary table
    if results:
        summary_rows = []
        for r in results:
            current_price = r['latest_price'] or np.nan
            fair_base = r['fair_base'] or np.nan
            fair_base_plus = r['fair_base_plus_div'] or np.nan
            fair_real = r['fair_real'] or np.nan
            fair_real_plus = r['fair_real_plus_div'] or np.nan

            ratio_base = fair_base / current_price if current_price and current_price > 0 else np.nan
            ratio_base_plus = fair_base_plus / current_price if current_price and current_price > 0 else np.nan
            ratio_real = fair_real / current_price if current_price and current_price > 0 else np.nan
            ratio_real_plus = fair_real_plus / current_price if current_price and current_price > 0 else np.nan

            row = {
                'Ticker': r['ticker'],
                'Market Price': current_price,
                'Price Currency': r.get('price_currency'),
                'DCF Fair Price (Base)': fair_base,
                'DCF Fair Price (Base + Div)': fair_base_plus,
                'Fair/Current (Base)': ratio_base,
                'Fair/Current (Base+Div)': ratio_base_plus,
                'DCF Fair Price (Inflation-Adj)': fair_real,
                'DCF Fair Price (Inflation-Adj + Div)': fair_real_plus,
                'Fair/Current (Inflation-Adj)': ratio_real,
                'Fair/Current (Inflation-Adj+Div)': ratio_real_plus,
                'Avg Div per Share': r['div_per_share'],
                'Market Cap': r['market_cap'],
                'Balance Sheet Currency FX used': r.get('fx_rate_used')
            }
            summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        st.subheader("DCF Summary (currency-normalised; base & inflation-adjusted; with and without dividends)")
        st.dataframe(df_summary)

        # detailed view
        for r in results:
            st.markdown(f"---\n### {r['ticker']}")
            col1, col2 = st.columns([2, 3])
            with col1:
                st.write("**Market data**")
                st.write({
                    'Price (normalized)': r['latest_price'],
                    'Price Currency': r.get('price_currency'),
                    'Price scale applied (raw*scale)': r.get('price_scale'),
                    'Market Cap': r['market_cap'],
                    'Shares Outstanding (est)': r['shares']
                })
                st.write("**Balance sheet (raw & normalized)**")
                st.write({
                    'Raw cash (as reported)': r['cash_raw'],
                    'Raw debt (as reported)': r['debt_raw'],
                    'Normalized cash (price currency)': r['cash'],
                    'Normalized debt (price currency)': r['debt'],
                    'FX rate used (bs->price currency)': r.get('fx_rate_used')
                })
                st.write("**Dividend estimates**")
                st.write({
                    'Avg dividend per share (historical)': r['div_per_share'],
                    'PV projected dividends (nominal, per share)': r['pv_div_nominal'],
                    'PV projected dividends (real/inflation-adjusted, per share)': r['pv_div_real']
                })

                st.write("**DCF results**")
                st.write({
                    'Enterprise value (base)': r['dcf_base']['enterprise_value'],
                    'Equity value (base)': r['dcf_base']['equity_value'],
                    'Fair price per share (base)': r['fair_base'],
                    'Fair price per share (base + dividends PV)': r['fair_base_plus_div']
                })
                st.write({
                    'Enterprise value (inflation-adj)': r['dcf_real']['enterprise_value'],
                    'Equity value (inflation-adj)': r['dcf_real']['equity_value'],
                    'Fair price per share (inflation-adj)': r['fair_real'],
                    'Fair price per share (inflation-adj + dividends PV)': r['fair_real_plus_div']
                })

            with col2:
                st.write("**Projected FCF (per year, base projection)**")
                proj_years_list = list(range(1, proj_years + 1))
                fig, ax = plt.subplots()
                ax.plot(proj_years_list, r['dcf_base']['fcf_forecast'], marker='o')
                ax.set_xlabel('Year')
                ax.set_ylabel('Projected FCF')
                ax.set_title(f"Projected FCF — {r['ticker']}")
                st.pyplot(fig)

                st.write("**Discounted cash flows (base)**")
                fig2, ax2 = plt.subplots()
                ax2.bar(proj_years_list, r['dcf_base']['discounted'])
                ax2.set_xlabel('Year')
                ax2.set_ylabel('Discounted FCF')
                ax2.set_title(f"Discounted FCF — {r['ticker']}")
                st.pyplot(fig2)

            if isinstance(r['historical_fcf'], pd.Series) and len(r['historical_fcf']) > 0:
                st.write("Historical Free Cash Flow")
                st.line_chart(r['historical_fcf'])

    else:
        st.warning("No successful valuations were computed.")

st.markdown("\n---\n*This tool attempts to normalise price and balance-sheet currencies for clearer valuations. Always verify currency mappings and FX conversions before making investment decisions.*")
