import numpy as np

class QAgent:
    def __init__(self, game, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_rate_min=0.01, num_episodes=100):
        # Initialize the Q-learning agent with a Q-table containing all zeros
        # where the rows represent states, columns represent actions, and the third dimension is for each action (Up, Down, Left, Right)
        self.q_table = np.zeros((game.maze_height, game.maze_width, 4)) # 4 actions: Up, Down, Left, Right
        self.learning_rate = learning_rate          # Learning rate controls how much the agent updates its Q-values after each action
        self.discount_factor = discount_factor      # Discount factor determines the importance of future rewards in the agent's decisions
        self.exploration_rate = exploration_rate  # Exploration rate determines the likelihood of the agent taking a random action
        self.exploration_rate_min = exploration_rate_min
        self.num_episodes = num_episodes
        self.goal_reward = 100
        self.wall_penalty = -10
        self.step_penalty = -1

    def get_exploration_rate(self, current_episode):
        exploration_rate = self.exploration_rate * (self.exploration_rate_min / self.exploration_rate) ** (current_episode / self.num_episodes)
        return exploration_rate

    def get_action(self, state, current_episode): # State is tuple representing where agent is in maze (x, y)
        exploration_rate = self.get_exploration_rate(current_episode)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[state]) # Choose the action with the highest Q-value for the given state

    def update_q_table(self, state, action, next_state, reward):
        # Find the best next action by selecting the action that maximizes the Q-value for the next state
        best_next_action = np.argmax(self.q_table[next_state])

        # Get the current Q-value for the current state and action
        current_q_value = self.q_table[state][action]

        # Q-value update using Q-learning formula
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - current_q_value)

        # Update the Q-table with the new Q-value for the current state and action
        self.q_table[state][action] = new_q_value

    def train_episode(self, current_episode, train=True):
        # Initialize the agent's current state to the maze's start position
        current_state = self.start_position
        is_done = False
        episode_reward = 0
        episode_step = 0
        path = [current_state]

        self.update_episode(current_episode)

        while not is_done:
            # Get the agent's action for the current state using its Q-table
            action = self.get_action(current_state, current_episode)

            # Compute the next state based on the chosen action
            next_state = (current_state[0] + self.actions[action][0], current_state[1] + self.actions[action][1])

            # Check if the next state is out of bounds or hitting a wall
            if next_state[0] < 0 or next_state[0] >= self.maze_height or next_state[1] < 0 or next_state[1] >= self.maze_width or self.maze[next_state[1]][next_state[0]] == 1:
                reward = self.wall_penalty
                next_state = current_state
            # Check if the agent reached the goal:
            elif next_state == (self.end_position):
                path.append(current_state)
                reward = self.goal_reward
                is_done = True
            # The agent takes a step but hasn't reached the goal yet
            else:
                path.append(current_state)
                reward = self.step_penalty

            # Update the cumulative reward and step count for the episode
            episode_reward += reward
            episode_step += 1

            # Update the agent's Q-table if training is enabled
            if train == True:
                self.update_q_table(current_state, action, next_state, reward)

            # Move to the next state for the next iteration
            current_state = next_state

            self.update(current_state, episode_step, episode_reward)
        
        return episode_reward, episode_step, path

    def train_agent(self, num_episodes=100):
        # Lists to store the data for plotting
        episode_rewards = []
        episode_steps = []

        # Loop over the specified number of episodes
        for episode in range(num_episodes):
            episode_reward, episode_step, path = self.train_episode(episode, train=True)

            # Store the episode's cumulative reward and the number of steps taken in their respective lists
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)

    def test_agent(self, num_episodes=1):
        # Simulate the agent's behavior in the maze for the specified number of episodes
        episode_reward, episode_step, path = self.train_episode(num_episodes, train=False)

        # Print the learned path of the agent
        print("Learned Path:")
        for row, col in path:
            print(f"({row}, {col})-> ", end='')
        print("Goal!")

        print("Number of steps:", episode_step)
        print("Total reward:", episode_reward)

        return episode_step, episode_reward
