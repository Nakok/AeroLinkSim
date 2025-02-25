import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 15)
ax.set_ylim(0, 10)
ax.set_title("Drone Formation Animation with Unique Flight Patterns")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# Number of drones and their initial positions (V-shape)
drone_positions = np.array([[5, 2], [4, 3], [6, 3], [3, 4], [7, 4]])
drone_dots, = ax.plot([], [], 'ro', markersize=8)

# Create unique movement patterns for each drone
patterns = [
    # Center drone: Moves in a gentle sine wave with a slight upward drift
    lambda t: (5 + np.sin(t / 10) * 2, 2 + t * 0.02),
    
    # Left drone: Moves in a zigzag pattern
    lambda t: (4 + np.sin(t / 5) * 1.5, 3 + t * 0.02),
    
    # Right drone: Moves in a circular path
    lambda t: (6 + np.cos(t / 15) * 1.5, 3 + np.sin(t / 15) * 1.5 + t * 0.02),
    
    # Far-left drone: Moves in a spiral pattern
    lambda t: (3 + np.cos(t / 20) * (1 + t / 300), 4 + np.sin(t / 20) * (1 + t / 300) + t * 0.02),
    
    # Far-right drone: Moves in a figure-eight pattern
    lambda t: (7 + np.sin(t / 15) * 2, 4 + np.sin(t / 10) * 1.5 + t * 0.02)
]

# Function to initialize the animation
def init():
    drone_dots.set_data([], [])
    return drone_dots,

# Function to update the animation
def update(frame):
    new_positions = np.array([pattern(frame) for pattern in patterns])
    drone_dots.set_data(new_positions[:, 0], new_positions[:, 1])
    return drone_dots,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=300, init_func=init, blit=True, interval=50)

# Save as GIF using PillowWriter
writer = PillowWriter(fps=20)
ani.save("drone_formation_unique_patterns.gif", writer=writer)

# Display the animation
plt.show()
