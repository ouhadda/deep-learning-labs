import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for displaying plots
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode for smooth animation

class GridWorldEnv:
    def __init__(self):
        self.size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]

        # action space: up, down, left, right
        self.action_map = {
            0: (-1, 0),   # up
            1: (1, 0),    # down
            2: (0, -1),   # left
            3: (0, 1),    # right
        }
        
        # Track cell rewards and visited cells
        self.cell_rewards = np.zeros((self.size, self.size))
        self.visited_cells = set()

        self.reset()

    def reset(self):
        self.agent_pos = list(self.start)
        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_pos)

    def step(self, action):
        move = self.action_map[action]
        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]

        # --- WALL CHECK ---
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            # hit wall → try a random valid direction
            valid_moves = []
            for a in range(4):
                m = self.action_map[a]
                nx = self.agent_pos[0] + m[0]
                ny = self.agent_pos[1] + m[1]
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    (nx, ny) not in self.obstacles):
                    valid_moves.append(a)
            
            if valid_moves:
                random_action = valid_moves[np.random.randint(0, len(valid_moves))]
                return self.step(random_action)  # Recursively step with random valid move
            else:
                # No valid moves available (trapped) - stay and get heavy penalty
                reward = -20
                return self._get_state(), reward, False, {}

        # --- OBSTACLE CHECK ---
        if (new_x, new_y) in self.obstacles:
            # hit obstacle → try a random valid direction
            valid_moves = []
            for a in range(4):
                m = self.action_map[a]
                nx = self.agent_pos[0] + m[0]
                ny = self.agent_pos[1] + m[1]
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    (nx, ny) not in self.obstacles):
                    valid_moves.append(a)
            
            if valid_moves:
                random_action = valid_moves[np.random.randint(0, len(valid_moves))]
                return self.step(random_action)  # Recursively step with random valid move
            else:
                # No valid moves available (trapped) - stay and get heavy penalty
                reward = -20
                return self._get_state(), reward, False, {}

        # --- MOVE (always moves to a valid cell) ---
        self.agent_pos = [new_x, new_y]
        reward = -1  # default step penalty for moving

        # --- GOAL CHECK ---
        if tuple(self.agent_pos) == self.goal:
            reward = +10
            return self._get_state(), reward, True, {}

        return self._get_state(), reward, False, {}

    # ========== RENDERING WITH MATPLOTLIB ==========
    def render(self, values=None, score=0, path=None):
        plt.clf()  # Clear previous figure
        
        # Create a white background grid
        grid = np.ones((self.size, self.size)) * 255  # White background
        
        ax = plt.gca()
        ax.imshow(grid, cmap="gray", interpolation="nearest", vmin=0, vmax=255)

        # Draw grid lines
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color="black", linewidth=2)
            ax.axvline(i - 0.5, color="black", linewidth=2)

        # Draw optimal path if provided
        if path is not None:
            for i in range(len(path) - 1):
                y1, x1 = path[i]
                y2, x2 = path[i + 1]
                ax.plot([x1, x2], [y1, y2], color="green", linewidth=3, alpha=0.7, zorder=1)

        # Draw reward values in each cell
        if values is not None:
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) not in self.obstacles and (i, j) != self.goal and (i, j) != self.start:
                        ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", 
                               fontsize=11, color="darkblue", weight="bold")

        # Draw obstacles (black with X)
        for (x, y) in self.obstacles:
            circle = plt.Circle((y, x), 0.3, color="black", zorder=3)
            ax.add_patch(circle)
            ax.text(y, x, "X", ha="center", va="center", fontsize=16, color="white", weight="bold", zorder=4)

        # Draw start (cyan S with reward)
        circle = plt.Circle((self.start[1], self.start[0]), 0.3, color="cyan", zorder=3)
        ax.add_patch(circle)
        ax.text(self.start[1], self.start[0], "S", ha="center", va="center", 
               fontsize=16, color="black", weight="bold", zorder=4)
        if values is not None:
            ax.text(self.start[1], self.start[0] - 0.35, f"{values[self.start[0], self.start[1]]:.1f}", 
                   ha="center", va="top", fontsize=9, color="darkblue")

        # Draw goal (blue G with reward)
        circle = plt.Circle((self.goal[1], self.goal[0]), 0.3, color="blue", zorder=3)
        ax.add_patch(circle)
        ax.text(self.goal[1], self.goal[0], "G", ha="center", va="center", 
               fontsize=16, color="white", weight="bold", zorder=4)
        if values is not None:
            ax.text(self.goal[1], self.goal[0] - 0.35, f"{values[self.goal[0], self.goal[1]]:.1f}", 
                   ha="center", va="top", fontsize=9, color="darkblue")

        # Draw agent (red A)
        if path is None:  # Only show agent when not displaying final path
            circle = plt.Circle((self.agent_pos[1], self.agent_pos[0]), 0.25, color="red", zorder=5)
            ax.add_patch(circle)
            ax.text(self.agent_pos[1], self.agent_pos[0], "A", ha="center", va="center", 
                   fontsize=14, color="white", weight="bold", zorder=6)

        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(self.size - 0.5, -0.5)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if path is None:
            plt.title(f"Grid World - Score: {score}", fontsize=16, weight="bold")
        else:
            plt.title(f"Optimal Path Found!", fontsize=16, weight="bold")
        
        plt.tight_layout()
        plt.pause(0.5)  # Pause for 0.5 seconds to see each step
