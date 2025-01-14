# -*- coding: cp1252 -*-
# generated by GPT.
# run from prompt: streamlit run sp500_backtest.py
# backtest2.py plots +/- 1.5 std. Sidebar inputs removed for phone use.
#
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up the Streamlit interface
st.title("Stock Index Backtest & Monte Carlo Simulation")
st.write("This app simulates the backtest performance and Monte Carlo projections for a given initial investment amount and term in years using historical data.")

# Move user input components to the sidebar
st.header("User Input")

# Dropdown menu for index selection
index_choice = st.selectbox(
    "Select Index:", 
    options=["S&P 500", "FTSE 100", "Nikkei", "Stoxx", "BSE", "Shanghai Composite Index"]
)

# Input from the user for initial amount and term
initial_amount = st.number_input("Enter Initial Investment Amount (�):", min_value=100.0, value=1000.0)
years = st.number_input("Enter Term (Years):", min_value=1, max_value=100, value=10)

# Fetch historical data based on the selected index
@st.cache_data
def fetch_data(ticker):
    index = yf.Ticker(ticker)
    data = index.history(period="max")
    data = data[['Close']]
    data.reset_index(inplace=True)
    return data

# Determine ticker symbol based on user selection
index_ticker_map = {
    "S&P 500": "^GSPC",
    "FTSE 100": "^FTSE",
    "Nikkei": "^N225",
    "Stoxx": "^STOXX50E",
    "BSE": "^BSESN",
    "Shanghai Composite Index": "000001.SS"
}

ticker_symbol = index_ticker_map[index_choice]

# Fetch data for the selected index
data = fetch_data(ticker_symbol)

# Calculate the backtest performance
def calculate_backtest(data, initial_amount, years):
    if years * 252 > len(data):
        st.warning("Insufficient data for the specified term. Reduce the term in years.")
        return None

    # Calculate yearly performance
    data['Year'] = data['Date'].dt.year
    yearly_data = data.groupby('Year').last().reset_index()

    # Calculate performance for the last 'years' term
    term_start_year = yearly_data['Year'].max() - years
    term_data = yearly_data[yearly_data['Year'] > term_start_year]

    if len(term_data) < years:
        st.warning("Insufficient data for the specified term. Reduce the term in years.")
        return None

    # Calculate cumulative returns
    start_price = term_data.iloc[0]['Close']
    end_price = term_data.iloc[-1]['Close']
    cumulative_return = (end_price / start_price) - 1
    final_amount = initial_amount * (1 + cumulative_return)

    return cumulative_return, final_amount, term_data

# Monte Carlo Simulation Function
# Original Monte Carlo Simulation Function
def original_monte_carlo_simulation(data, initial_amount, years, num_simulations=100):
    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()
    daily_returns = data['Daily Return'].dropna()

    # Mean and standard deviation of daily returns to simulate realistic movement
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    # Run standard Monte Carlo simulations
    simulations = []
    days = years * 252  # Approximate trading days in the term

    for _ in range(num_simulations):
        simulated_prices = [initial_amount]
        
        for _ in range(days):
            # Simulate daily price change using random normal values
            daily_return = np.random.normal(mu, sigma)
            next_price = simulated_prices[-1] * (1 + daily_return)
            # Replace any negative value with zero
            next_price = max(next_price, 1)
            simulated_prices.append(next_price)

        simulations.append(simulated_prices)

    # Convert to DataFrame for analysis
    simulation_df = pd.DataFrame(simulations).T
    median_simulation = simulation_df.median(axis=1)
    std_dev = simulation_df.std(axis=1)
    
    # Calculate �1.5 standard deviation
    upper_bound = median_simulation + 1.5 * std_dev
    lower_bound = median_simulation - 1.5 * std_dev

    # Ensure lower bound does not go below zero
    lower_bound = lower_bound.clip(lower=0)

    # Extract yearly values for the table
    yearly_indices = [(i + 1) * 252 for i in range(years)]
    yearly_median = median_simulation.iloc[yearly_indices].values
    yearly_upper = upper_bound.iloc[yearly_indices].values
    yearly_lower = lower_bound.iloc[yearly_indices].values

    return median_simulation, upper_bound, lower_bound, yearly_median, yearly_upper, yearly_lower

#Markov chain with random walk or Metropolis hastings
def markov_chain_monte_carlo(data, initial_amount, years, num_simulations=100):
    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()
    daily_returns = data['Daily Return'].dropna()

    # Mean and standard deviation of daily returns to simulate realistic movement
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    # Run MCMC simulations
    simulations = []
    days = years * 252  # Approximate trading days in the term

    for _ in range(num_simulations):
        simulated_prices = [initial_amount]
        current_return = mu

        for _ in range(days):
            # Simulate a new return based on the current return with a small random change
            change = np.random.normal(loc=0, scale=sigma)
            current_return += change
            # Apply the Markov process: the next price depends on the current price
            next_price = simulated_prices[-1] * (1 + current_return)
            # Replace any negative value with zero
            next_price = max(next_price, 1)
            simulated_prices.append(next_price)

        simulations.append(simulated_prices)

    # Convert to DataFrame for analysis
    simulation_df = pd.DataFrame(simulations).T
    median_simulation = simulation_df.median(axis=1)
    std_dev = simulation_df.std(axis=1)
    
    # Calculate �1.5 standard deviation
    upper_bound = median_simulation + 1.5 * std_dev
    lower_bound = median_simulation - 1.5 * std_dev

    # Ensure lower bound does not go below zero
    lower_bound = lower_bound.clip(lower=0)

    # Extract yearly values for the table
    yearly_indices = [(i + 1) * 252 for i in range(years)]
    yearly_median = median_simulation.iloc[yearly_indices].values
    yearly_upper = upper_bound.iloc[yearly_indices].values
    yearly_lower = lower_bound.iloc[yearly_indices].values

    return median_simulation, upper_bound, lower_bound, yearly_median, yearly_upper, yearly_lower

# Improved Markov chain
def improved_markov_chain_monte_carlo(data, initial_amount, years, num_simulations=100, mean_reversion_strength=0.03, volatility_cap=0.02):
    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()
    daily_returns = data['Daily Return'].dropna()

    # Mean and standard deviation of daily returns to simulate realistic movement
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    # Run MCMC simulations
    simulations = []
    days = years * 252  # Approximate trading days in the term

    for _ in range(num_simulations):
        simulated_prices = [initial_amount]
        current_return = mu

        for _ in range(days):
            # Simulate a new return based on the current return with a small random change
            change = np.random.normal(loc=0, scale=min(sigma * 0.5, volatility_cap))  # Cap the change to prevent extremes
            # Introduce mean reversion effect
            reversion = mean_reversion_strength * (mu - current_return)
            
            # Limit the effect of change and reversion
            current_return += max(min(change + reversion, volatility_cap), -volatility_cap)

            # Apply the Markov process: the next price depends on the current price
            next_price = simulated_prices[-1] * (1 + current_return)
            # Replace any negative value with zero
            next_price = max(next_price, 0)
            simulated_prices.append(next_price)

        simulations.append(simulated_prices)

    # Convert to DataFrame for analysis
    simulation_df = pd.DataFrame(simulations).T
    median_simulation = simulation_df.median(axis=1)
    std_dev = simulation_df.std(axis=1)
    
    # Calculate �1.5 standard deviation
    upper_bound = median_simulation + 1.5 * std_dev
    lower_bound = median_simulation - 1.5 * std_dev

    # Ensure lower bound does not go below zero
    lower_bound = lower_bound.clip(lower=0)

    # Extract yearly values for the table
    yearly_indices = [(i + 1) * 252 for i in range(years)]
    yearly_median = median_simulation.iloc[yearly_indices].values
    yearly_upper = upper_bound.iloc[yearly_indices].values
    yearly_lower = lower_bound.iloc[yearly_indices].values

    return median_simulation, upper_bound, lower_bound, yearly_median, yearly_upper, yearly_lower


# Markov Geometric Brownian motion
def gbm_monte_carlo_simulation(data, initial_amount, years, num_simulations=100, adjustment_factor=1.02):
    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()
    daily_returns = data['Daily Return'].dropna()

    # Calculate mean and standard deviation of daily returns
    mu = daily_returns.mean() * adjustment_factor  # Slightly adjust the mean upward
    sigma = daily_returns.std()

    # Run GBM-based Monte Carlo simulations
    simulations = []
    days = years * 252  # Approximate trading days in the term
    dt = 1 / 252  # Time step for daily returns

    for _ in range(num_simulations):
        simulated_prices = [initial_amount]

        for _ in range(days):
            # Use GBM formula: dS = mu * S * dt + sigma * S * dW
            dW = np.random.normal(0, np.sqrt(dt))
            drift = (mu - 0.5 * sigma ** 2) * dt  # Adjusted drift
            diffusion = sigma * dW
            daily_return = np.exp(drift + diffusion)
            
            # Next price
            next_price = simulated_prices[-1] * daily_return
            # Replace any negative value with zero
            next_price = max(next_price, 0)
            simulated_prices.append(next_price)

        simulations.append(simulated_prices)

    # Convert to DataFrame for analysis
    simulation_df = pd.DataFrame(simulations).T
    median_simulation = simulation_df.median(axis=1)
    std_dev = simulation_df.std(axis=1)
    
    # Calculate �1.5 standard deviation
    upper_bound = median_simulation + 1.5 * std_dev
    lower_bound = median_simulation - 1.5 * std_dev

    # Ensure lower bound does not go below zero
    lower_bound = lower_bound.clip(lower=0)

    # Extract yearly values for the table
    yearly_indices = [(i + 1) * 252 for i in range(years)]
    yearly_median = median_simulation.iloc[yearly_indices].values
    yearly_upper = upper_bound.iloc[yearly_indices].values
    yearly_lower = lower_bound.iloc[yearly_indices].values

    return median_simulation, upper_bound, lower_bound, yearly_median, yearly_upper, yearly_lower


# Dropdown menu for simulation selection
simulation_type = st.selectbox(
    "Select Monte Carlo Simulation Type:",
    options=["Original Monte Carlo", "Random Walk MCMC", "Geometric Brownian Motion MCMC"]
)

# Perform calculation and display results
if st.button("Run Backtest"):
    result = calculate_backtest(data, initial_amount, years)
    
    if result:
        cumulative_return, final_amount, term_data = result
        st.write(f"**Initial Amount:** �{initial_amount:,.2f}")
        st.write(f"**Term:** {years} years")
        st.write(f"**Cumulative Return:** {cumulative_return * 100:.2f}%")
        st.write(f"**Final Amount:** �{final_amount:,.2f}")

        # Plot the performance on the main page
        fig, ax = plt.subplots()
        ax.plot(term_data['Year'], term_data['Close'], marker='o', linestyle='-')
        ax.set_title(f"{index_choice} Performance Over the Last {years} Years")
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{index_choice} Index Level")
        st.pyplot(fig)

# Run Monte Carlo Simulation based on user choice
if st.button("Run Monte Carlo Simulation"):
    if simulation_type == "Original Monte Carlo":
        median_sim, upper_bound, lower_bound, yearly_median, yearly_upper, yearly_lower = original_monte_carlo_simulation(
            data, initial_amount, years)
    elif simulation_type == "Random Walk MCMC":
        median_sim, upper_bound, lower_bound, yearly_median, yearly_upper, yearly_lower = improved_markov_chain_monte_carlo(data, initial_amount, years)
    elif simulation_type == "Geometric Brownian Motion MCMC":
        median_sim, upper_bound, lower_bound, yearly_median, yearly_upper, yearly_lower = gbm_monte_carlo_simulation(
            data, initial_amount, years)

    # Plot the simulation results on the main page
    fig, ax = plt.subplots()
    ax.plot(median_sim, label='Median Simulation', color='blue')
    ax.fill_between(range(len(median_sim)), lower_bound, upper_bound, color='lightgray', alpha=0.5, label='�1.5 Std Dev')
    ax.set_title(f"Monte Carlo Simulation - {years} Years ({index_choice})")
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    st.pyplot(fig)

    # Create a DataFrame for the yearly results and format it to 2 decimal places
    year_labels = [f"Year {i+1}" for i in range(years)]
    result_df = pd.DataFrame({
        "Year": year_labels,
        "Median ($)": yearly_median,
        "Median + 1.5 Std Dev ($)": yearly_upper,
        "Median - 1.5 Std Dev ($)": yearly_lower
    })

    result_df["Median ($)"] = result_df["Median ($)"].map('{:,.2f}'.format)
    result_df["Median + 1.5 Std Dev ($)"] = result_df["Median + 1.5 Std Dev ($)"].map('{:,.2f}'.format)
    result_df["Median - 1.5 Std Dev ($)"] = result_df["Median - 1.5 Std Dev ($)"].map('{:,.2f}'.format)

    # Display the results in a table on the main page
    st.write("### Yearly Projected Portfolio Value")
    st.table(result_df)
