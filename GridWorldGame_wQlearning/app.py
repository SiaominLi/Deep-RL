from flask import Flask, render_template, request, jsonify
import numpy as np
import random
from tqdm import tqdm

app = Flask(__name__)


class GridWorld:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n))
        self.start = None
        self.end = None
        self.obstacles = []
        self.Q = {}
        for i in range(n):
            for j in range(n):
                if (i, j) not in self.obstacles and (i, j) != self.end:
                    self.Q[(i, j)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

    def set_start(self, row, col):
        self.start = (row, col)

    def set_end(self, row, col):
        self.end = (row, col)

    def set_obstacle(self, row, col):
        self.obstacles.append((row, col))
        self.Q.pop((row, col), None)  # Remove obstacles from Q-table

    def is_reachable(self, state):
        """Check if a state is valid (within bounds and not an obstacle)."""
        row, col = state
        return 0 <= row < self.n and 0 <= col < self.n and state not in self.obstacles

    def get_next_state(self, state, action):
        """Return the next state based on current action."""
        row, col = state
        if action == 'up' and self.is_reachable((row - 1, col)):
            return (row - 1, col)
        elif action == 'down' and self.is_reachable((row + 1, col)):
            return (row + 1, col)
        elif action == 'left' and self.is_reachable((row, col - 1)):
            return (row, col - 1)
        elif action == 'right' and self.is_reachable((row, col + 1)):
            return (row, col + 1)
        return state  # Remain in the same state if action is not valid

    def get_reward(self, state, action):
        """Return reward for a given action."""
        next_state = self.get_next_state(state, action)
        if next_state == self.end:
            return 10  # terminal reward
        if next_state in self.obstacles:
            return -1  # obstacle penalty
        return -0.01  # step penalty

    def update_Q(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        """Update the Q-value for a given state and action."""
        max_future_q = max(self.Q[next_state].values()) if next_state != self.end else 0
        current_q = self.Q[state][action]
        new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
        self.Q[state][action] = new_q

    def choose_action(self, state, epsilon=0.1):
        """Choose an action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < epsilon:
            return random.choice(['up', 'down', 'left', 'right'])  # Explore
        else:
            return max(self.Q[state], key=self.Q[state].get)  # Exploit

    def learn(self, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
        """Run Q-learning algorithm."""
        for _ in range(episodes):
            state = self.start
            while state != self.end and state not in self.obstacles:
                action = self.choose_action(state, epsilon)
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(state, action)
                self.update_Q(state, action, reward, next_state, alpha, gamma)
                state = next_state

    def print_Q(self):
        """Prints the Q-table."""
        print("Q-table:")
        for state, actions in self.Q.items():
            print(f"State {state}: {actions}")

    def get_optimal_path(self):
        """Trace the optimal path from start to end based on learned Q-values."""
        path = [self.start]
        state = self.start
        while state != self.end:
            action = max(self.Q[state], key=self.Q[state].get)
            state = self.get_next_state(state, action)
            if state in path:  # Check for loops
                break
            path.append(state)
        return path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_grid', methods=['POST'])
def generate_grid():
    n = int(request.form['n'])
    if n < 3 or n > 7:
        return "Error: n should be between 3 and 7"
    return render_template('index.html', n=n)

@app.route('/evaluate_policy', methods=['POST'])
def evaluate_policy():
    points = request.json['points']
    grid_size = int(request.json['n'])

    grid_world = GridWorld(grid_size)

    for point in points:
        row, col, cell_type = point['row'], point['col'], point['type']

        if cell_type == 'start':
            grid_world.set_start(row, col)
        elif cell_type == 'end':
            grid_world.set_end(row, col)
        elif cell_type == 'obstacle':
            grid_world.set_obstacle(row, col)

    grid_world.learn()
    optimal_path = grid_world.get_optimal_path()

    return jsonify({
        'optimal_path': optimal_path
    })

if __name__ == '__main__':
    app.run(port=9000, debug=True)
