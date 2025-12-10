# Q-Learning Agent - Much smarter than random!
from gridworld_env import GridWorldEnv
import numpy as np
import time
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.env = env
        self.lr = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: state (row, col) -> action -> Q-value
        self.q_table = {}
        self.initialize_q_table()
        
        self.cumulative_reward = 0
        self.path = []
        self.episode_rewards = []
        
    def initialize_q_table(self):
        """Initialize Q-table for all states"""
        for i in range(self.env.size):
            for j in range(self.env.size):
                if (i, j) not in self.env.obstacles:
                    self.q_table[(i, j)] = {0: 0, 1: 0, 2: 0, 3: 0}  # 4 actions
    
    def get_best_action(self, state):
        """Get best action from Q-table (greedy)"""
        if state not in self.q_table:
            return np.random.randint(0, 4)
        return np.argmax(list(self.q_table[state].values()))
    
    def get_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)  # Explore
        else:
            return self.get_best_action(state)  # Exploit
    
    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning formula"""
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # Q-learning update
        max_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train(self, episodes=50):
        """Train the agent for multiple episodes"""
        print("Training Q-Learning Agent...")
        
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            episode_path = [state]
            
            for step in range(100):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-table
                self.update_q_value(state, action, reward, next_state)
                
                episode_reward += reward
                steps += 1
                episode_path.append(next_state)
                
                if done:
                    print(f"Episode {episode + 1}: Reached goal in {steps} steps, Reward: {episode_reward}")
                    break
                
                state = next_state
            
            self.episode_rewards.append(episode_reward)
            
            # Gradually reduce exploration
            if episode % 10 == 0:
                self.epsilon *= 0.9
        
        print(f"\nTraining complete! Average reward (last 10 episodes): {np.mean(self.episode_rewards[-10:]):.2f}")
    
    def run_greedy(self, render=True):
        """Run the agent using learned policy (no exploration)"""
        print("\nRunning with learned policy...")
        state = self.env.reset()
        self.path = [state]
        self.cumulative_reward = 0
        
        for step in range(100):
            action = self.get_best_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            self.cumulative_reward += reward
            self.path.append(next_state)
            
            print(f"[{step}] State={next_state}, Action={action}, Reward={reward}, Total: {self.cumulative_reward}")
            
            # Convert Q-table to value function for visualization
            values = np.zeros((self.env.size, self.env.size))
            for (i, j), actions in self.q_table.items():
                values[i, j] = max(actions.values())
            
            if render:
                self.env.render(values=values, score=self.cumulative_reward)
            
            if done:
                print(f"Goal reached! Final score: {self.cumulative_reward}")
                break
            
            state = next_state
        
        # Show final path
        print(f"\nOptimal path found: {self.path}")
        values = np.zeros((self.env.size, self.env.size))
        for (i, j), actions in self.q_table.items():
            values[i, j] = max(actions.values())
        
        self.env.render(values=values, path=self.path)
        plt.show()


if __name__ == "__main__":
    env = GridWorldEnv()
    agent = QLearningAgent(env, learning_rate=0.15, discount_factor=0.95, epsilon=0.3)
    
    # Train the agent
    agent.train(episodes=50)
    
    # Run the trained agent
    agent.run_greedy(render=True)
