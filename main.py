import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Function to download and process data
def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Return'] = data['Adj Close'].pct_change()
    data['Volatility'] = data['Return'].rolling(window=20).std() * np.sqrt(252)
    data.dropna(inplace=True)
    return data

# Black-Scholes model
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Monte Carlo simulation
def monte_carlo_simulation(S0, T, r, sigma, n_simulations=1000, n_steps=252):
    dt = T/n_steps
    price_paths = np.zeros((n_steps + 1, n_simulations))
    price_paths[0] = S0
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        price_paths[t] = price_paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return price_paths

# RL Trading Agent class
class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))  # Changed lr to learning_rate
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = np.random.choice(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



# Streamlit UI
def main():
    st.title("Algorithmic Trading with Reinforcement Learning")
    
    # Sidebar for selecting stock and parameters
    st.sidebar.header("Stock Selection")
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2021-01-01"))
    r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.02)
    sigma = st.sidebar.slider("Volatility", 0.1, 0.5, 0.2)
    
    # Fetching data
    data = download_data(ticker, start_date, end_date)
    st.write(f"### {ticker} Data")
    st.line_chart(data['Adj Close'])
    
    # Running Monte Carlo Simulation
    if st.sidebar.button("Run Simulation"):
        st.write("### Monte Carlo Simulation")
        n_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000)
        price_paths = monte_carlo_simulation(data['Adj Close'].iloc[-1], 1, r, sigma, n_simulations)
        st.line_chart(price_paths)

        # Placeholder for RL agent
        state_size = len(data.columns)
        action_size = 3  # buy, sell, hold
        agent = TradingAgent(state_size, action_size)

        # Training the agent (you can add your logic here)
        st.write("Training the agent and visualizing results here...")

if __name__ == "__main__":
    main()
