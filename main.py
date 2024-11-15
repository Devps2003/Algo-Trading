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
from scipy.stats import norm

def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes Delta.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return -norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")


# Monte Carlo simulation
def monte_carlo_simulation1(S0, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    price_paths = np.zeros((n_simulations, n_steps + 1))
    price_paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        )
    return price_paths
def monte_carlo_simulation2(S0, T, r, sigma, n_simulations=1000, n_steps=252):
    dt = T/n_steps
    price_paths = np.zeros((n_steps + 1, n_simulations))
    price_paths[0] = S0
    
    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)
        price_paths[t] = price_paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return price_paths



def hedging_simulation(S0, K, T, r, sigma, n_simulations, n_steps):
    dt = T / n_steps
    time_steps = np.linspace(0, T, n_steps + 1)
    price_paths = monte_carlo_simulation1(S0, T, r, sigma, n_simulations, n_steps)

    hedge_pnl = np.zeros(n_simulations)
    
    for sim in range(n_simulations):
        cash = 0
        hedge = 0
        for t in range(n_steps):
            St = price_paths[sim, t]
            d1 = (np.log(St / K) + (r + 0.5 * sigma ** 2) * (T - time_steps[t])) / (sigma * np.sqrt(T - time_steps[t]))
            delta = norm.cdf(d1)  # Cumulative distribution function of standard normal
            
            # Adjust portfolio
            dS = price_paths[sim, t + 1] - St
            cash += hedge * dS - (delta - hedge) * St * (r * dt)
            hedge = delta

        hedge_pnl[sim] = cash + hedge * price_paths[sim, -1] - max(price_paths[sim, -1] - K, 0)
    
    return hedge_pnl, price_paths



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
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        for index in minibatch:
            state, action, reward, next_state, done = self.memory[index]
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def calculate_metrics(price_paths):
    strategy_returns = price_paths[-1] / price_paths[0] - 1

    # Cumulative Return
    cumulative_return = np.mean(strategy_returns)
    
    # Sharpe Ratio
    risk_free_rate = 0.02  # Example risk-free rate
    excess_returns = strategy_returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(strategy_returns)
    
    # Maximum Drawdown
    cumulative_returns = np.cumprod(1 + strategy_returns)
    drawdowns = cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1
    max_drawdown = np.min(drawdowns)
    
    return cumulative_return, sharpe_ratio, max_drawdown

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
    
    if st.sidebar.button("Run Hedging Strategy"):
        st.write("### Hedging Simulation")
        
        # Ensure the value is a float or int
        current_price = float(data['Adj Close'].iloc[-1])
        
        n_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000)
        price_paths = monte_carlo_simulation1(current_price, 1, r, sigma, n_simulations, 50)
        
        K = st.sidebar.slider(
            "Strike Price",
            0.8 * current_price,
            1.2 * current_price,
            current_price
        )
        
        n_steps = st.sidebar.slider("Number of Steps", 50, 500, 252)
        
        with st.spinner("Running hedging simulation..."):
            hedge_pnl, price_paths = hedging_simulation(current_price, K, 1, r, sigma, n_simulations, n_steps)
        
        st.success("Hedging simulation completed!")
        
        # Visualize results
        st.write("### Hedging P&L")
        st.line_chart(hedge_pnl)

        st.write("### Price Paths")
        st.line_chart(price_paths)

    # Running Monte Carlo Simulation
    if st.sidebar.button("Run Simulation"):
        st.write("### Monte Carlo Simulation")
        n_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000)
        price_paths = monte_carlo_simulation2(data['Adj Close'].iloc[-1], 1, r, sigma, n_simulations,50)
        st.line_chart(price_paths)

        # Calculate and display performance metrics
        cumulative_return, sharpe_ratio, max_drawdown = calculate_metrics(price_paths)
        
        st.write("### Performance Metrics")
        st.metric("Cumulative Return", f"{cumulative_return:.2%}")
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        st.metric("Maximum Drawdown", f"{max_drawdown:.2%}")

        # RL agent integration
        state_size = price_paths.shape[0]
        action_size = 3  # buy, sell, hold
        agent = TradingAgent(state_size, action_size)

        # Simulated training data
        batch_size = 32
        states = price_paths.T

        for episode in range(10):  # Example: 10 episodes for training
            state = states[0].reshape(1, -1)
            total_reward = 0

            for t in range(1, len(states)):
                action = agent.act(state)
                next_state = states[t].reshape(1, -1)
                reward = states[t][-1] - states[t-1][-1] if action == 1 else 0
                total_reward += reward
                done = (t == len(states) - 1)
                
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                
                state = next_state

            st.write(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # Hedging strategy
        


if __name__ == "__main__":
    main()
