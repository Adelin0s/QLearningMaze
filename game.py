import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from agent import *

class Game:
    def __init__(self, start_position, end_position):
        # Initialize Maze object with the provided maze, start_position, and goal position
        self.agent = None
        self.maze = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ])

        # Actions the agent can take: Up, Down, Left, Right. Each action is represented as a tuple of two values: (row_change, column_change)
        self.actions = [
            (-1,  0), # Up: Moving one step up, reducing the row index by 1
            ( 1,  0),  # Down: Moving on step down, increasing the row index by 1
            ( 0, -1), # Left: Moving one step to the left, reducing the column index by 1
            ( 0,  1)   # Right: Moving one step to the right, increasing the column index by 1
        ]

        self.maze_height = self.maze.shape[0] # Get the height of the maze (number of rows)
        self.maze_width = self.maze.shape[1]  # Get the width of the maze (number of columns)
        self.start_position = start_position    # Set the start position in the maze as a tuple (x, y)
        self.end_position = end_position      # Set the goal position in the maze as a tuple (x, y)
        self.robot_position = start_position

    def initialize(self):
        self.agent = QAgent(self)
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.imshow(self.maze, cmap='gray')
        self.episode_text = self.ax.text(0.1, -0.1, '', transform=self.ax.transAxes, ha='center')
        self.steps_text = self.ax.text(0.4, -0.1, '', transform=self.ax.transAxes, ha='center')
        self.reward_text = self.ax.text(0.7, -0.1, '', transform=self.ax.transAxes, ha='center')

        self.ax.text(self.start_position[0], self.start_position[1], 'S', ha='center', va='center', color='red', fontsize=12)
        self.ax.text(self.end_position[0], self.end_position[1], 'G', ha='center', va='center', color='green', fontsize=12)
        plt.xticks([]), plt.yticks([])
        base_length = 0.1
        height = 0.25
        triangle_coords = [
            (self.start_position[0], self.start_position[1]),  # Punct de bază
            (self.start_position[0] + base_length, self.start_position[1]),  # Punct de bază
            (self.start_position[0] + base_length / 2, self.start_position[1] + height)  # Vârf
        ]
        #plt.scatter(self.start_position[0], self.en[1] + 1, marker='^', color='blue', s=100)
        self.robot_triangle = Polygon(triangle_coords, closed=True, edgecolor='yellow', facecolor='yellow')
        self.ax.add_patch(self.robot_triangle)

    def update(self, position, steps, reward):
        base_length = 0.1
        height = 0.25
        triangle_coords = [
            (position[0], position[1]),  
            (position[0] + base_length, position[1]),
            (position[0] + base_length / 2, position[1] + height)
        ]
        self.robot_triangle.set_xy(triangle_coords)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.steps_text.set_text(f"Steps: {steps}")
        self.reward_text.set_text(f"Reward: {reward}")

    def update_episode(self, new_episode):
        self.episode_text.set_text(f"Episode {new_episode}")

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
            action = self.agent.get_action(current_state, current_episode)

            # Compute the next state based on the chosen action
            next_state = (current_state[0] + self.actions[action][0], current_state[1] + self.actions[action][1])

            # Check if the next state is out of bounds or hitting a wall
            if next_state[0] < 0 or next_state[0] >= self.maze_height or next_state[1] < 0 or next_state[1] >= self.maze_width or self.maze[next_state[1]][next_state[0]] == 1:
                reward = self.agent.wall_penalty
                next_state = current_state
            # Check if the agent reached the goal:
            elif next_state == (self.end_position):
                path.append(current_state)
                reward = self.agent.goal_reward
                is_done = True
            # The agent takes a step but hasn't reached the goal yet
            else:
                path.append(current_state)
                reward = self.agent.step_penalty

            # Update the cumulative reward and step count for the episode
            episode_reward += reward
            episode_step += 1

            # Update the agent's Q-table if training is enabled
            if train == True:
                self.agent.update_q_table(current_state, action, next_state, reward)

            # Move to the next state for the next iteration
            current_state = next_state

            self.update(current_state, episode_step, episode_reward)
            plt.pause(0.1)  # Pause for visualization
        
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



if __name__ == "__main__":
    game = Game((0, 0), (4, 4))
    game.initialize()
    game.train_agent()
    
# def train_agent(agent, game, num_episodes=100):
#     # Lists to store the data for plotting
#     episode_rewards = []
#     episode_steps = []

#     # Loop over the specified number of episodes
#     for episode in range(num_episodes):
#         episode_reward, episode_step, path = train_episode(agent, game, episode, train=True)

#         # Store the episode's cumulative reward and the number of steps taken in their respective lists
#         episode_rewards.append(episode_reward)
#         episode_steps.append(episode_step)

# train_agent(agent, game, num_episodes=100)
# test_agent(agent, game, num_episodes=100)