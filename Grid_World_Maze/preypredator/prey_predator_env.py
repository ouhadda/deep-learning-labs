import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.ion()

class PreyPredatorEnv:
    def __init__(self, size=10, num_preys=3):
        self.size = size
        self.num_preys = num_preys
        self.obstacles = [(3, 3), (3, 4), (4, 3), (5, 5), (6, 6), (7, 7)]
        
        # Safe haven (prey can hide here)
        self.safe_haven = (9, 9)
        
        # Multiple preys
        self.preys = [self.random_position() for _ in range(num_preys)]
        
        # Single predator
        self.predator_pos = self.random_position()
        self.predator_last_pos = self.predator_pos  # Track to prevent back-and-forth
        
        # Actions: up, down, left, right
        self.action_map = {
            0: (-1, 0),   # up
            1: (1, 0),    # down
            2: (0, -1),   # left
            3: (0, 1),    # right
        }
        
        self.step_count = 0
        self.prey_caught = 0
        self.prey_escaped = 0
        self.prey_safe = 0
    
    def random_position(self):
        """Get a random valid position"""
        while True:
            pos = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if pos not in self.obstacles:
                return pos
    
    def move_entity(self, entity_pos, action):
        """Move entity in a direction"""
        move = self.action_map[action]
        new_x = entity_pos[0] + move[0]
        new_y = entity_pos[1] + move[1]
        
        # Boundary check
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            return entity_pos
        
        # Obstacle check
        if (new_x, new_y) in self.obstacles:
            return entity_pos
        
        return (new_x, new_y)
    
    def distance(self, pos1, pos2):
        """Manhattan distance"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_prey_action(self, prey_pos):
        """Prey AI: Move away from predator or towards safe haven"""
        dist_to_predator = self.distance(prey_pos, self.predator_pos)
        dist_to_haven = self.distance(prey_pos, self.safe_haven)
        
        # If in danger, prioritize safe haven
        if dist_to_predator < 6:
            # Move towards safe haven
            best_action = None
            min_dist = float('inf')
            
            for action in range(4):
                new_pos = self.move_entity(prey_pos, action)
                new_dist = self.distance(new_pos, self.safe_haven)
                if new_dist < min_dist:
                    min_dist = new_dist
                    best_action = action
            
            return best_action if best_action is not None else np.random.randint(0, 4)
        else:
            # Random wandering if safe
            return np.random.randint(0, 4)
    
    def get_predator_action(self):
        """Predator AI: Chase nearest prey (intelligent, avoids back-and-forth)"""
        # Find nearest prey
        nearest_prey = min(self.preys, key=lambda p: self.distance(self.predator_pos, p))
        
        best_action = None
        min_dist = float('inf')
        
        for action in range(4):
            new_pos = self.move_entity(self.predator_pos, action)
            new_dist = self.distance(new_pos, nearest_prey)
            
            # Prefer moving forward, avoid going back to last position (prevents back-and-forth)
            if new_pos == self.predator_last_pos:
                continue  # Skip this action
            
            if new_dist < min_dist:
                min_dist = new_dist
                best_action = action
        
        # If all actions lead back, pick the best one anyway
        if best_action is None:
            for action in range(4):
                new_pos = self.move_entity(self.predator_pos, action)
                new_dist = self.distance(new_pos, nearest_prey)
                if best_action is None or new_dist < self.distance(self.move_entity(self.predator_pos, best_action), nearest_prey):
                    best_action = action
        
        return best_action if best_action is not None else np.random.randint(0, 4)
    
    def step(self):
        """Simulate one step of the game"""
        self.step_count += 1
        
        # Move all preys (AI controlled)
        for i in range(len(self.preys)):
            prey_action = self.get_prey_action(self.preys[i])
            self.preys[i] = self.move_entity(self.preys[i], prey_action)
        
        # Store predator position before moving (to prevent back-and-forth)
        self.predator_last_pos = self.predator_pos
        
        # Move predator (AI controlled)
        predator_action = self.get_predator_action()
        self.predator_pos = self.move_entity(self.predator_pos, predator_action)
        
        prey_caught_this_step = False
        prey_safe_this_step = False
        
        # Check each prey
        for i, prey_pos in enumerate(self.preys):
            # Check if prey reached safe haven
            if prey_pos == self.safe_haven:
                self.prey_safe += 1
                prey_safe_this_step = True
                self.preys[i] = self.random_position()
            # Check if prey caught by predator
            elif self.distance(prey_pos, self.predator_pos) == 0:
                self.prey_caught += 1
                prey_caught_this_step = True
                self.preys[i] = self.random_position()
        
        return prey_caught_this_step, prey_safe_this_step
    
    def render(self):
        """Render the game"""
        plt.clf()
        
        # Create grid background
        grid = np.ones((self.size, self.size)) * 255
        
        ax = plt.gca()
        ax.imshow(grid, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        
        # Draw grid lines
        for i in range(self.size + 1):
            ax.axhline(i - 0.5, color="lightgray", linewidth=1)
            ax.axvline(i - 0.5, color="lightgray", linewidth=1)
        
        # Draw safe haven (blue zone)
        haven_circle = plt.Circle((self.safe_haven[1], self.safe_haven[0]), 0.4, color="cyan", alpha=0.7, zorder=2)
        ax.add_patch(haven_circle)
        ax.text(self.safe_haven[1], self.safe_haven[0], "S", ha="center", va="center", 
               fontsize=12, color="darkblue", weight="bold", zorder=3)
        
        # Draw obstacles
        for (x, y) in self.obstacles:
            circle = plt.Circle((y, x), 0.3, color="black", zorder=3)
            ax.add_patch(circle)
            ax.text(y, x, "X", ha="center", va="center", fontsize=12, color="white", weight="bold", zorder=4)
        
        # Draw all preys (green)
        for prey_pos in self.preys:
            circle = plt.Circle((prey_pos[1], prey_pos[0]), 0.32, color="lime", zorder=5)
            ax.add_patch(circle)
            ax.text(prey_pos[1], prey_pos[0], "P", ha="center", va="center", 
                   fontsize=13, color="darkgreen", weight="bold", zorder=6)
        
        # Draw predator (red, single)
        predator_circle = plt.Circle((self.predator_pos[1], self.predator_pos[0]), 0.35, color="red", zorder=5)
        ax.add_patch(predator_circle)
        ax.text(self.predator_pos[1], self.predator_pos[0], "D", ha="center", va="center", 
               fontsize=14, color="white", weight="bold", zorder=6)
        
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(self.size - 0.5, -0.5)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        title = f"Prey-Predator Game | Step: {self.step_count} | Caught: {self.prey_caught} | Safe: {self.prey_safe} | Active Preys: {len(self.preys)}"
        plt.title(title, fontsize=14, weight="bold")
        plt.tight_layout()
        plt.pause(0.2)
