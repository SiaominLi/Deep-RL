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

    optimal_policy = grid_world.get_optimal_policy()

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

    return jsonify({'policy_arrows': policy_arrows})