from flask import Flask, render_template, request, jsonify
import numpy as np
import random

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

    def bellman_update(self, state, action, gamma=0.9):
        row, col = state
        next_states = []
        transition_probs = []
        if action == 'up':
            next_state = (row - 1, col)
            if self.is_reachable(next_state):
                next_states.append(next_state)
                transition_probs.append(1.0)
        elif action == 'down':
            next_state = (row + 1, col)
            if self.is_reachable(next_state):
                next_states.append(next_state)
                transition_probs.append(1.0)
        elif action == 'left':
            next_state = (row, col - 1)
            if self.is_reachable(next_state):
                next_states.append(next_state)
                transition_probs.append(1.0)
        elif action == 'right':
            next_state = (row, col + 1)
            if self.is_reachable(next_state):
                next_states.append(next_state)
                transition_probs.append(1.0)

        max_value = -float('inf')
        for i, next_state in enumerate(next_states):
            if next_state == self.end:
                value = transition_probs[i] * 10  # terminal reward
            elif next_state in self.obstacles:
                value = transition_probs[i] * -1  # obstacle penalty
            else:
                value = transition_probs[i] * (gamma * self.value_function.get(next_state, 0) - 0.01)  # step penalty

            if value > max_value:
                max_value = value

        self.value_function[state] = max_value

    def is_reachable(self, state):
        # 使用广度优先搜索判断状态是否可达终止状态
        queue = [self.end]
        visited = set()
        while queue:
            curr_state = queue.pop(0)
            if curr_state in visited:
                continue
            visited.add(curr_state)
            row, col = curr_state
            neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
            for neighbor in neighbors:
                if neighbor == state:
                    return True
                if neighbor not in self.obstacles and 0 <= neighbor[0] < self.n and 0 <= neighbor[1] < self.n:
                    queue.append(neighbor)
        return False

    def value_iteration(self, gamma=0.9, epsilon=1e-6):
        """Perform value iteration to converge the value function."""
        self.initialize_random_policy()
        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for state in self.policy:
                if state != self.end:
                    action = self.policy[state]
                    old_value = self.value_function.get(state, 0)
                    self.bellman_update(state, action, gamma)
                    new_value = self.value_function[state]
                    delta = max(delta, abs(old_value - new_value))
                    print(f"Iteration: State {state}, Value = {new_value}")  # 打印当前状态值

        print("\nValue Iteration Converged!")
        print("Optimal Value Function:")
        for state, value in self.value_function.items():
            print(f"State {state}: Value = {value}")

        optimal_policy = self.get_optimal_policy()
        print("\nOptimal Policy:")
        for state, action in optimal_policy.items():
            print(f"State {state}: Action = {action}")
            
    def get_optimal_policy(self):
        """Derive the optimal policy from the converged value function."""
        optimal_policy = {}
        for state in self.policy:
            if state != self.end:
                max_value = -float('inf')
                best_action = None
                for action in self.get_possible_actions(state):
                    next_states = []
                    transition_probs = []
                    if action == 'up':
                        next_states.append((state[0] - 1, state[1]))
                        transition_probs.append(1.0)
                    elif action == 'down':
                        next_states.append((state[0] + 1, state[1]))
                        transition_probs.append(1.0)
                    elif action == 'left':
                        next_states.append((state[0], state[1] - 1))
                        transition_probs.append(1.0)
                    elif action == 'right':
                        next_states.append((state[0], state[1] + 1))
                        transition_probs.append(1.0)

                    value = sum(transition_probs[i] * self.value_function.get(next_state, 0) for i, next_state in enumerate(next_states))
                    if value > max_value:
                        max_value = value
                        best_action = action
                optimal_policy[state] = best_action
        return optimal_policy
    def get_optimal_path(self):
        """根据最优策略从起点出发找到到达终点的路径"""
        path = []
        curr_state = self.start
        while curr_state != self.end:
            path.append(curr_state)
            if curr_state not in self.policy:
                # 无法到达终点
                return []
            action = self.policy[curr_state]
            if action == 'up':
                curr_state = (curr_state[0] - 1, curr_state[1])
            elif action == 'down':
                curr_state = (curr_state[0] + 1, curr_state[1])
            elif action == 'left':
                curr_state = (curr_state[0], curr_state[1] - 1)
            elif action == 'right':
                curr_state = (curr_state[0], curr_state[1] + 1)
        path.append(self.end)
        return path
    def find_optimal_path(self, max_iterations=1000, gamma=0.9, epsilon=1e-6):
        """自动重新初始化策略并继续寻找,直到找到从起点到终点的路径"""
        iteration = 0
        while iteration < max_iterations:
            self.initialize_random_policy()
            self.value_iteration(gamma, epsilon)
            optimal_path = self.get_optimal_path()
            if optimal_path:
                return optimal_path
            iteration += 1
        return []  # 如果达到最大迭代次数仍未找到路径,返回空列表
    def get_possible_actions(self, state):
        """Get the possible actions from a given state."""
        actions = []
        row, col = state
        if row > 0 and (row - 1, col) not in self.obstacles:
            actions.append('up')
        if row < self.n - 1 and (row + 1, col) not in self.obstacles:
            actions.append('down')
        if col > 0 and (row, col - 1) not in self.obstacles:
            actions.append('left')
        if col < self.n - 1 and (row, col + 1) not in self.obstacles:
            actions.append('right')
        return actions


@app.route('/')
def index():
    return render_template('index.html', n=7)  # 修改这一行，传递合适的网格尺寸值给模板

@app.route('/evaluate_policy', methods=['POST'])
def evaluate_policy():
    points = request.json['points']
    n = int(request.json['n'])

    grid_world = GridWorld(n)

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

    if not optimal_path:
        return jsonify({'error': 'Cannot find a path from the start state to the end state after maximum iterations.'})

    policy_arrows = []
    for row in range(n):
        for col in range(n):
            if (row, col) in optimal_policy:
                action = optimal_policy[(row, col)]
                if action == 'up':
                    policy_arrows.append('↑')
                elif action == 'down':
                    policy_arrows.append('↓')
                elif action == 'left':
                    policy_arrows.append('←')
                elif action == 'right':
                    policy_arrows.append('→')
            else:
                policy_arrows.append('')

    return jsonify({'policy_arrows': policy_arrows, 'optimal_path': optimal_path})

if __name__ == '__main__':
    app.run(debug=True)
