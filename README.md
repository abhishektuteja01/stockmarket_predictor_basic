# Stock Market Predictor: Q-Learning-Based Trading Agent

This project builds a Q-learning-based stock trading simulator for a **single stock**, using historical stock data and technical indicators to train an intelligent trading agent. The system includes data acquisition, environment simulation, reinforcement learning, and visualization of trading performance.

---

## 🚀 Features

- Download and process historical stock price data using `yfinance`
- Compute technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) using the `ta` library
- Simulate a trading environment with Buy/Hold/Sell actions
- Train a Q-learning agent to maximize long-term returns
- Evaluate against a baseline Buy & Hold strategy
- Interactive input for selecting the stock ticker
- Visualizations of learning performance and final asset value

---

## 🧠 Technologies Used

- Python
- Pandas
- yfinance
- ta (Technical Analysis library)
- Matplotlib
- Custom environment and agent logic

---

## 📁 Folder Structure

```
stockmarket_predictor_basic/
├── data_prep.py              # Download + preprocess stock data
├── trading_env.py            # Custom trading environment class
├── q_learning.py             # Q-learning agent + training loop
├── README.md                 # Project documentation
```

---

## 🧪 How to Set It Up

### 1. Clone the Repo
```bash
git clone <https://github.com/abhishektuteja01/stockmarket_predictor_basic.git>
cd stockmarket_predictor_basic
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install yfinance ta pandas matplotlib
```

---

## ▶️ How to Run the Project

Run this project in two steps:

### Step 1: Prepare the Data
```bash
python data_prep.py
```
This will download and process AAPL stock data and save it as `AAPL_processed.csv`.

### Step 2: Train the Q-Learning Agent
```bash
python q_learning.py
```
This will:
- Load `AAPL_processed.csv`
- Run Q-learning training for 1000 episodes
- Display a performance graph
- Print final portfolio performance vs. Buy & Hold strategy

---

## 📝 What Each File Does

- `./data_prep.py`  
  Downloads historical AAPL stock data, computes technical indicators using the `ta` library, and outputs `AAPL_processed.csv`.

- `./trading_env.py`  
  Simulates a trading environment, defines buy/sell/hold actions, and computes portfolio value and reward after each action.

- `./q_learning.py`  
  Contains the Q-learning agent and training loop, loads the processed dataset, trains the agent, and visualizes performance.

---

## 📊 Example Output

The system produces a line chart comparing:

- Total reward earned in each episode
- Final portfolio value across episodes

It also prints the Buy & Hold return over the same time period for comparison.

---

## 🔧 Future Improvements

- Add support for multiple stocks or portfolio allocation
- Use deep reinforcement learning (e.g. DQN, Policy Gradients)
- Incorporate news or sentiment data
- Improve state abstraction and feature engineering

---

## 👨‍💻 Author

Abhishek Tuteja — Foundations of AI (CS 5100)

---

## 🪪 License

This project is licensed under the [MIT License](./LICENSE).
