#!/usr/bin/env python3
"""
Prey-Predator Autonomous Game
- Multiple Preys (green P) try to escape to the safe zone (S)
- Single Predator (red D) chases the preys intelligently
- Preys caught = go back to random position
- Preys reach safe haven = saved and respawn
- Predator doesn't get stuck in back-and-forth movement
"""

from prey_predator_env import PreyPredatorEnv
import matplotlib.pyplot as plt
import time

def main():
    env = PreyPredatorEnv(size=10, num_preys=3)
    
    print("=" * 70)
    print("PREY-PREDATOR AUTONOMOUS GAME")
    print("=" * 70)
    print("Green P = Preys (escape to safe zone)")
    print("Red D = Predator (single, intelligent)")
    print("Cyan S = Safe Haven (preys are saved here!)")
    print("X = Obstacles")
    print("=" * 70)
    
    try:
        for episode in range(5):  # Run 5 episodes
            print(f"\n--- Episode {episode + 1} ---")
            env.step_count = 0
            
            for step in range(300):  # Max 300 steps per episode
                prey_caught, prey_safe = env.step()
                
                env.render()
                
                if prey_caught:
                    print(f"Step {env.step_count}: Prey caught! (Total caught: {env.prey_caught})")
                if prey_safe:
                    print(f"Step {env.step_count}: Prey reached safety! (Total safe: {env.prey_safe})")
                
                time.sleep(0.1)
        
        print(f"\n{'=' * 70}")
        print(f"GAME SUMMARY")
        print(f"Total Preys Caught: {env.prey_caught}")
        print(f"Total Preys Saved: {env.prey_safe}")
        total = env.prey_caught + env.prey_safe
        if total > 0:
            print(f"Predator Success Rate: {env.prey_caught / total * 100:.1f}%")
            print(f"Preys Survival Rate: {env.prey_safe / total * 100:.1f}%")
        print(f"{'=' * 70}")
        
        plt.show()
    
    except KeyboardInterrupt:
        print("\n\nGame stopped by user")

if __name__ == "__main__":
    main()

