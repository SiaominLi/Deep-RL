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
        self.policy = {}
        self.value_function = {}
        # New attribute for action logging
        self.action_log = []

    def set_start(self, row, col):
        self.start = (row, col)

    def set_end(self, row, col):
        self.end = (row, col)

    def set_obstacle(self, row, col):
        self.obstacles.append((row, col))

    def initialize_random_policy(self):
        """Initialize a random policy for all states."""
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) not in self.obstacles and (i, j) != self.end:
                    self.policy[(i, j)] = random.choice(['up', 'down', 'left', 'right'])

    def is_reachable(self, state):
        """Check if a state is valid (within bounds and not an obstacle)."""
        row, col = state
        return 0 <= row < self.n and 0 <= col < self.n and state not in self.obstacles

    def bellman_update(self, state, action, gamma=0.9):
        """Perform a Bellman update for a given state and action."""

        is_valid = False
        row, col = state

        if action == 'up' and self.is_reachable((row - 1, col)):
            next_state = (row - 1, col)
        elif action == 'down' and self.is_reachable((row + 1, col)):
            next_state = (row + 1, col)
        elif action == 'left' and self.is_reachable((row, col - 1)):
            next_state = (row, col - 1)
        elif action == 'right' and self.is_reachable((row, col + 1)):
            next_state = (row, col + 1)
        else:
            next_state = state  # Remain in the same state if the action is not valid

        # Compute the new value function for the state
        if next_state == self.end:
            reward = 10  # terminal reward
        elif next_state in self.obstacles:
            reward = -1  # obstacle penalty
        else:
            reward = gamma * self.value_function.get(next_state, 0) - 0.01  # step penalty

        self.value_function[state] = reward

    def value_iteration(self, gamma=0.9, epsilon=1e-6, max_iterations=1000):
        """Perform value iteration to converge the value function."""
        self.initialize_random_policy()
        delta = float('inf')
        iterations = 0
        while delta > epsilon and iterations < max_iterations:
            iterations += 1
            delta = 0
            for state in self.policy:
                if state != self.end:
                    action = self.policy[state]
                    old_value = self.value_function.get(state, 0)
                    self.bellman_update(state, action, gamma)
                    new_value = self.value_function[state]
                    delta = max(delta, abs(old_value - new_value))
            
        # Print the value function in the specified format after each iteration
        # self.print_value_function()

    def print_value_function(self):
        """Prints the value function in a matrix format."""
        value_matrix = np.zeros((self.n, self.n))
        for state, value in self.value_function.items():
            value_matrix[state] = value
        print("Value Function:")
        print(value_matrix)

    def get_optimal_path(self):
        """根据最优策略从起点出发找到到达终点的路径，包含防止无限循环的措施。"""
        path = []
        visited = set()  # 追踪已訪問的狀態
        curr_state = self.start
        steps = 0
        max_steps = self.n * self.n  # 設置合理的步數上限

        while curr_state != self.end and steps < max_steps:
            if curr_state in visited:
                # print("Detected loop, stopping path search.")
                return []  # 如果检测到循环，返回空列表
            visited.add(curr_state)
            path.append(curr_state)
            action = self.policy.get(curr_state)

            # 根据行动更新当前状态
            if action == 'up':
                next_state = (curr_state[0] - 1, curr_state[1])
            elif action == 'down':
                next_state = (curr_state[0] + 1, curr_state[1])
            elif action == 'left':
                next_state = (curr_state[0], curr_state[1] - 1)
            elif action == 'right':
                next_state = (curr_state[0], curr_state[1] + 1)
            else:
                # print("No valid action found, stopping path search.")
                return []  # 如果没有有效行动，返回空列表

            self.action_log.append({
                "state": list(curr_state),
                "action": action,
                "reward": self.value_function[curr_state]
            })

            # 檢查是否出界或走進障礙物
            if not self.is_reachable(next_state):
                # print("Next state is not reachable, stopping path search.")
                return []  # 如果下一状态不可达，返回空列表

            curr_state = next_state
            steps += 1

        if curr_state == self.end:
            self.action_log.append({
                "state": list(curr_state),
                "action": '',
                "reward": 0
            })
            path.append(self.end)
            return path
        else:
            # print("Failed to reach end within step limit.")
            return []  # 如果在步数限制内未到达终点，返回空列表

    def find_optimal_path(self, max_iterations=100000, gamma=0.9, epsilon=1e-6):
        """自動重新初始化策略並繼續尋找,直到找到從起點到終點的最短路徑或達到嘗試上限"""
        best_path = []
        best_length = float('inf')
        for _ in tqdm(range(max_iterations)):
            self.initialize_random_policy()
            self.value_iteration(gamma, epsilon)
            self.action_log = []  # Reset the action log for each iteration
            path = self.get_optimal_path()
            if path and (len(path) < best_length):
                best_path = path
                best_length = len(path)
                best_actions = self.action_log.copy()  # Copy current log as it's the best one yet
        if best_path:
            self.action_log = best_actions  # Store only the log of the best path found
            print("Optimal Path Found:", best_path)
            self.print_value_function()
        else:
            print("Failed to find a path after maximum iterations")
        return best_path

    def get_action_log(self):
        """Retrieve the action log."""
        return self.action_log


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

    grid_world.value_iteration()
    optimal_path = grid_world.find_optimal_path()
    action_log = grid_world.get_action_log()

    new_optimal_path = ''
    for point in optimal_path:
        new_optimal_path += '[' + str(point[0]) + ', ' + str(point[1]) + '] → '

    new_optimal_path = new_optimal_path[:-3]

    if not new_optimal_path:
        new_optimal_path = 'Optimal Path Not Found'

    print(f'ANS: {new_optimal_path}')

    # print('LEN:')
    # print(len(action_log))

    return jsonify({
        'optimal_path': optimal_path,  # Send the list of coordinates directly
        'action_log': action_log[-1:]
    })


if __name__ == '__main__':
    app.run(port=9000, debug=True)