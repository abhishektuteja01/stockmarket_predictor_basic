import random
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = defaultdict(lambda: [0.0 for _ in actions])
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(round(state[k], 2) if isinstance(state[k], (float, int)) else 0 for k in sorted(state))

    def select_action(self, state):
        key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.q_table[key].index(max(self.q_table[key]))

    def update(self, state, action, reward, next_state):
        s = self.get_state_key(state)
        s_next = self.get_state_key(next_state)

        max_next_q = max(self.q_table[s_next])
        td_target = reward + self.gamma * max_next_q
        td_delta = td_target - self.q_table[s][action]

        self.q_table[s][action] += self.lr * td_delta


# Training loop and setup
import pandas as pd
from trading_env import TradingEnv

if __name__ == '__main__':
    # Load historical stock data with indicators
    data = pd.read_csv("AAPL_processed.csv", header=0)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])

    # Initialize environment and agent
    env = TradingEnv(data)
    agent = QLearningAgent(actions=[0, 1, 2])  # 0 = Hold, 1 = Buy, 2 = Sell

    num_episodes = 1000  # Increased from 10 to 1000

    episode_rewards = []
    final_assets = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        # Print progress every 100 episodes to avoid flooding console
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Final Asset = {env.get_total_asset():.2f}")
        
        episode_rewards.append(total_reward)
        final_assets.append(env.get_total_asset())

    # Plot performance
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Agent Rewards")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(final_assets, label='Final Asset Value')
    plt.axhline(y=env.initial_balance, color='r', linestyle='--', label='Initial Balance')
    plt.xlabel("Episode")
    plt.ylabel("Asset Value ($)")
    plt.title("Q-Learning Agent Asset Performance")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Show final performance
    print(f"\nFinal Performance after {num_episodes} episodes:")
    print(f"Starting Balance: ${env.initial_balance:.2f}")
    print(f"Final Asset Value: ${final_assets[-1]:.2f}")
    print(f"Total Return: {(final_assets[-1] - env.initial_balance) / env.initial_balance:.2%}")

    # Baseline: Buy & Hold return
    buy_price = data.iloc[0]['Close']
    sell_price = data.iloc[-1]['Close']
    buy_hold_return = (sell_price - buy_price) / buy_price
    print(f"\nBuy & Hold Return: {buy_hold_return:.2%}")