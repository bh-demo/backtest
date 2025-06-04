import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import base64  # for image display.

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
    
# Set page config
st.set_page_config(
    page_title="Stock Correlation Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .title {
        text-align: center;
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stDownloadButton>button {
        background-color: #008CBA;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

add_logo()
# App title
st.title('📈 Stock Correlation Analyzer')
st.markdown("""
Analyze correlations between stocks over different time periods. 
Get started by entering stock symbols in the sidebar.
""")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # Stock symbols input
    default_symbols = ['ISF.L', 'SPY', 'GBPUSD=X', 'BATS.L','DHYG.L', 'UESD.L', 'FLOT.L', 'TI5G.L', 'JEPG.L']
    symbols_input = st.text_area(
        "Enter stock symbols (comma or space separated)", 
        value=", ".join(default_symbols))
    
    # Parse symbols
    symbols = [s.strip().upper() for s in symbols_input.replace(',', ' ').split() if s.strip()]
    
    # Start year selection
    start_year = st.slider(
        "Start year for data (will use earliest available common date)",
        min_value=1990,
        max_value=datetime.now().year - 1,
        value=2016
    )
    
    # Correlation windows
    windows = st.multiselect(
        "Select correlation windows (days)",
        options=[20, 60, 120, 240, 500],
        default=[60, 120]
    )
    
    # Submit button
    analyze_button = st.button("Analyze Stocks")

# Main app function
def analyze_stocks(symbols, start_year, windows):
    """Main analysis function"""
    
    # Display status
    with st.spinner(f"Fetching data for {len(symbols)} stocks from {start_year}..."):
        # Define the start date
        start_date = f"{start_year}-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download stock data
        data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
        
        # If only one symbol is provided, yfinance returns a different format
        if len(symbols) == 1:
            data = data.rename(columns={col: f"{symbols[0]}_{col}" for col in data.columns})
            data = pd.DataFrame({symbols[0]: data[f"{symbols[0]}_Adj Close"]})
        else:
            # For multiple symbols, extract adjusted close prices
            adj_close = pd.DataFrame()
            for symbol in symbols:
                if symbol in data.columns.get_level_values(0):
                    adj_close[symbol] = data[symbol]['Close']
                else:
                    st.warning(f"No data found for symbol {symbol}")
            
            data = adj_close
        
        # Drop symbols that have no data
        data = data.dropna(axis=1, how='all')
        if data.empty:
            st.error("No data available for any of the provided symbols.")
            return None, None
        
        # Find the latest start date among all stocks
        common_start_date = data.apply(lambda x: x.first_valid_index()).max()
        
        # Filter data to only include dates after the common start date
        processed_data = data.loc[common_start_date:]
        
        # Calculate daily percentage changes
        pct_changes = processed_data.pct_change().dropna()
        
        return pct_changes, common_start_date.date()

def plot_correlation_heatmap(corr_matrix, title):
    """Plot a heatmap of the correlation matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title(title, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)

# Run analysis when button is clicked
if analyze_button and symbols:
    # Perform analysis
    pct_changes, common_start_date = analyze_stocks(symbols, start_year, windows)
    
    if pct_changes is not None:
        # Display results header
        st.success(f"Analysis completed for {len(pct_changes.columns)} stocks with data from {common_start_date} to {pct_changes.index[-1].date()}")
        st.write(f"Total data points: {len(pct_changes)}")
        
        # Calculate and display standard deviations
        st.subheader("Standard Deviations (Volatility pct)")
        std_devs = 100*pct_changes.std(ddof=1)  # Using ddof=1 for sample standard deviation
        std_devs_df = pd.DataFrame(std_devs, columns=['Standard Deviation'])
        std_devs_df = std_devs_df.sort_values('Standard Deviation', ascending=False)  # Sort descending
        st.dataframe(std_devs_df.style.background_gradient(cmap='viridis').format("{:.4f}"))
        
        # Download button for standard deviations
        csv_std = std_devs_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Standard Deviations (CSV)",
            data=csv_std,
            file_name='stock_standard_deviations.csv',
            mime='text/csv',
            key="download_std"
        )
        
        # Show raw data option
        with st.expander("View Raw Data"):
            st.dataframe(pct_changes)
            
            # Download button for raw data
            csv = pct_changes.to_csv().encode('utf-8')
            st.download_button(
                label="Download Percentage Changes Data (CSV)",
                data=csv,
                file_name='stock_percentage_changes.csv',
                mime='text/csv'
            )
        
        # Calculate and display correlation matrices
        for window in sorted(windows):
            if len(pct_changes) < window:
                st.warning(f"Not enough data points ({len(pct_changes)}) for {window}-day window. Skipping.")
                continue
            
            st.subheader(f"{window}-Day Correlation Matrix")
            
            # Get the most recent 'window' days of data
            recent_data = pct_changes.iloc[-window:]
            
            # Calculate Pearson correlation matrix
            corr_matrix = recent_data.corr(method='pearson')
            
            # Display as table
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))
            
            # Plot heatmap
            #plot_correlation_heatmap(corr_matrix, f"{window}-Day Stock Correlations")
            
            # Download button for correlation matrix
            csv = corr_matrix.to_csv().encode('utf-8')
            st.download_button(
                label=f"Download {window}-Day Correlation Matrix (CSV)",
                data=csv,
                file_name=f'correlation_matrix_{window}days.csv',
                mime='text/csv',
                key=f"download_{window}"
            )
        
        # Show price trends
        st.subheader("Normalized Price Trends")
        normalized_prices = (pct_changes + 1).cumprod()
        fig, ax = plt.subplots(figsize=(12, 6))
        for column in normalized_prices.columns:
            ax.plot(normalized_prices.index, normalized_prices[column], label=column)
        ax.set_title("Normalized Price Trends (Starting at 1.0)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Price")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        st.pyplot(fig)
        
elif analyze_button and not symbols:
    st.warning("Please enter at least one stock symbol to analyze.")

# Add some info/help sections
with st.expander("How to use this app"):
    st.markdown("""
    1. **Enter stock symbols** in the sidebar (comma or space separated)
    2. **Select the start year** for historical data (will use earliest common date available)
    3. **Choose correlation windows** (in days) to analyze
    4. Click **"Analyze Stocks"** button
    5. View the correlation matrices and visualizations
    """)

with st.expander("About this analysis"):
    st.markdown("""
    - **Correlation** measures how stocks move in relation to each other (-1 to 1 scale)
    - **1.0** means perfect positive correlation (stocks move together)
    - **-1.0** means perfect negative correlation (stocks move opposite)
    - **0** means no correlation
    - **Standard Deviation** measures the volatility of each stock's returns
    - Analysis uses **percentage daily changes** in adjusted closing prices
    - Data sourced from Yahoo Finance via yfinance
    """)

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
    Stock Correlation Analyzer | Data from Yahoo Finance
    </div>
    """, unsafe_allow_html=True)
