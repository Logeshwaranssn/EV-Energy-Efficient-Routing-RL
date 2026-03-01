import numpy as np
import random

# =========================================================
#            EV ENERGY EFFICIENT ROUTE USING RL
# =========================================================

# ------------------ ROAD NETWORK (GRAPH) ------------------
# Each road: (next_node, distance_km, slope, traffic)

graph = {
    'A': [('B', 2, 1, 2), ('D', 4, 0, 1)],
    'B': [('A', 2, 1, 2), ('C', 2, 2, 3), ('E', 3, -1, 2)],
    'C': [('B', 2, 2, 3), ('F', 3, 1, 1)],
    'D': [('A', 4, 0, 1), ('E', 2, 1, 2)],
    'E': [('D', 2, 1, 2), ('B', 3, -1, 2), ('F', 2, 0, 1)],
    'F': []  # destination
}

START = 'A'
GOAL = 'F'


# ------------------ ENERGY MODEL ------------------
def energy_cost(distance, slope, traffic):
    base = 0.15 * distance        # base battery usage per km
    slope_cost = 0.05 * slope     # uphill uses more energy
    traffic_cost = 0.10 * traffic # traffic stop-go energy
    return base + slope_cost + traffic_cost


# ------------------ Q-LEARNING PARAMETERS ------------------
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.2   # exploration probability

Q = {}


def get_Q(state, action):
    if (state, action) not in Q:
        Q[(state, action)] = 0.0
    return Q[(state, action)]


# ------------------ ACTION SELECTION ------------------
def choose_action(state):
    # exploration
    if random.random() < epsilon:
        return random.choice(graph[state])

    # exploitation
    actions = graph[state]
    q_values = [get_Q(state, a[0]) for a in actions]
    return actions[np.argmax(q_values)]


# ------------------ TRAINING ------------------
episodes = 500

for ep in range(episodes):

    state = START
    battery = 10  # EV battery in kWh

    while state != GOAL and battery > 0:

        action = choose_action(state)
        next_state, dist, slope, traffic = action

        # energy consumption
        energy = energy_cost(dist, slope, traffic)

        # reward = negative energy (we want minimum energy)
        reward = -energy

        old_q = get_Q(state, next_state)

        # best future Q
        future_actions = graph[next_state]
        if future_actions:
            future_q = max([get_Q(next_state, a[0]) for a in future_actions])
        else:
            future_q = 0

        # Q-learning update
        new_q = old_q + alpha * (reward + gamma * future_q - old_q)
        Q[(state, next_state)] = new_q

        # battery consumption
        battery -= energy
        state = next_state


# ------------------ SHOW LEARNED ROUTE ------------------
print("\nLearned Energy Efficient Path:\n")

state = START
path = [state]

while state != GOAL:
    actions = graph[state]
    q_values = [get_Q(state, a[0]) for a in actions]
    best = actions[np.argmax(q_values)]
    state = best[0]
    path.append(state)

print(" -> ".join(path))


# ------------------ ENERGY CALCULATION OF FINAL PATH ------------------
total_energy = 0
state = START

while state != GOAL:
    actions = graph[state]
    q_values = [get_Q(state, a[0]) for a in actions]
    best = actions[np.argmax(q_values)]
    next_state, dist, slope, traffic = best

    total_energy += energy_cost(dist, slope, traffic)
    state = next_state

print(f"\nEstimated Energy Used: {round(total_energy,2)} kWh")
