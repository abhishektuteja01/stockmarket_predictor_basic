
class TradingEnv:
    def __init__(self, data, initial_balance=10000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_asset = self.balance
        self.trades = []
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        # Use only relevant features rather than the entire row
        state = {
            'price': row['Close'],
            'sma_20': row['SMA_20'],
            'rsi_14': row['RSI_14'],
            'macd': row['MACD'],
            'balance': self.balance,
            'shares_held': self.shares_held
        }
        return state

    def step(self, action):
        # 0 = Hold, 1 = Buy, 2 = Sell
        current_price = float(self.data.iloc[self.current_step]['Close'])
        
        # Store the current state for reward calculation
        prev_total_asset = self.balance + self.shares_held * current_price
        
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
                self.trades.append(('buy', self.current_step, current_price))
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                self.trades.append(('sell', self.current_step, current_price))
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate rewards based on price movement
        if not done:
            next_price = float(self.data.iloc[self.current_step]['Close'])
            # Calculate reward based on position and price movement
            if self.shares_held > 0:  # If holding shares
                reward = (next_price - current_price) * self.shares_held  # Reward/penalty for price change
            else:
                reward = 0  # No immediate reward when holding cash
        else:
            # Terminal reward based on overall performance
            reward = self.balance + self.shares_held * current_price - self.initial_balance
            
        self.total_asset = self.balance + self.shares_held * current_price
        
        return self._get_state(), reward, done

    def get_total_asset(self):
        return self.total_asset