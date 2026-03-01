
# ============================================================
#   EV ENERGY-EFFICIENT ROUTE USING REINFORCEMENT LEARNING
#   (FINAL VERSION – with Goal Reward & Learning Curve)
# ============================================================

import matplotlib
matplotlib.use("TkAgg")   # needed for clicking on Windows

import osmnx as ox
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# LOAD MAP
# ------------------------------------------------------------
print("Loading map...")
G = ox.graph_from_xml("map.osm", simplify=True)

# convert one-way streets to usable navigation network
G = ox.convert.to_undirected(G)

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

# ------------------------------------------------------------
# ADD TRAFFIC + SLOPE (simulation attributes)
# ------------------------------------------------------------
for u, v, k, data in G.edges(keys=True, data=True):
    data["traffic"] = random.randint(1, 5)
    data["slope"] = random.uniform(-3, 5)

# ------------------------------------------------------------
# ENERGY MODEL
# ------------------------------------------------------------
def energy_cost(edge):
    length_km = edge["length"] / 1000
    traffic = edge["traffic"]
    slope = edge["slope"]

    base = 0.18 * length_km
    traffic_cost = 0.03 * traffic
    slope_cost = 0.02 * max(slope, 0)

    return base + traffic_cost + slope_cost

# ------------------------------------------------------------
# CLICK START & DESTINATION
# ------------------------------------------------------------
print("\nClick START location on map window...")
fig, ax = ox.plot_graph(G, show=False, close=False)

start_click = plt.ginput(1)
START = ox.distance.nearest_nodes(G, start_click[0][0], start_click[0][1])
print("Start node:", START)

print("\nClick DESTINATION location on map window...")
dest_click = plt.ginput(1)
GOAL = ox.distance.nearest_nodes(G, dest_click[0][0], dest_click[0][1])
print("Goal node:", GOAL)

plt.close()

# ------------------------------------------------------------
# RL PARAMETERS
# ------------------------------------------------------------
alpha = 0.2
gamma = 0.9
epsilon = 0.9   # high exploration initially

Q = {}
episode_rewards = []

def get_Q(s, a):
    if (s, a) not in Q:
        Q[(s, a)] = 0
    return Q[(s, a)]

def choose_action(state):
    neighbors = list(G.neighbors(state))
    if not neighbors:
        return None

    if random.random() < epsilon:
        return random.choice(neighbors)

    q_vals = [get_Q(state, n) for n in neighbors]
    return neighbors[np.argmax(q_vals)]

# ------------------------------------------------------------
# TRAIN RL AGENT
# ------------------------------------------------------------
print("\nTraining RL agent...")

EPISODES = 2500

for ep in range(EPISODES):

    state = START
    battery = 5
    total_reward = 0
    visited = set()

    for step in range(400):

        if state == GOAL or battery <= 0:
            break

        action = choose_action(state)
        if action is None:
            break

        edge = list(G.get_edge_data(state, action).values())[0]
        energy = energy_cost(edge)

        # reward function
        reward = -energy

        # GOAL reward ⭐
        if action == GOAL:
            reward += 100

        # loop penalty
        if action in visited:
            reward -= 5
        visited.add(action)

        total_reward += reward

        old_q = get_Q(state, action)

        future_neighbors = list(G.neighbors(action))
        if future_neighbors:
            future_q = max(get_Q(action, n) for n in future_neighbors)
        else:
            future_q = 0

        Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

        battery -= energy
        state = action

        if state == GOAL:
            break

    episode_rewards.append(total_reward)

    # epsilon decay (agent becomes smarter)
    epsilon = max(0.05, epsilon * 0.995)

print("Training finished")

# ------------------------------------------------------------
# EXTRACT RL PATH
# ------------------------------------------------------------
path = [START]
state = START

for _ in range(600):
    if state == GOAL:
        break

    neighbors = list(G.neighbors(state))
    if not neighbors:
        break

    q_vals = [get_Q(state, n) for n in neighbors]
    state = neighbors[np.argmax(q_vals)]
    path.append(state)

print("RL path nodes:", len(path))

# ------------------------------------------------------------
# SHORTEST PATH
# ------------------------------------------------------------
shortest = nx.shortest_path(G, START, GOAL, weight="length")
print("Shortest path nodes:", len(shortest))

# ------------------------------------------------------------
# ENERGY COMPARISON
# ------------------------------------------------------------
def path_energy(route):
    total = 0
    for i in range(len(route)-1):
        edge = list(G.get_edge_data(route[i], route[i+1]).values())[0]
        total += energy_cost(edge)
    return total

rl_energy = path_energy(path)
short_energy = path_energy(shortest)

print("\nEnergy Comparison:")
print("RL Route Energy:", round(rl_energy,3), "kWh")
print("Shortest Route Energy:", round(short_energy,3), "kWh")

# ------------------------------------------------------------
# ROUTE VISUALIZATION
# ------------------------------------------------------------
print("\nDrawing routes...")

fig, ax = ox.plot_graph(G, node_size=5, edge_color="gray",
                        show=False, close=False)

ox.plot_graph_route(G, shortest, route_color="red",
                    route_linewidth=4, ax=ax, show=False, close=False)

ox.plot_graph_route(G, path, route_color="blue",
                    route_linewidth=3, ax=ax, show=False, close=False)

plt.title("Blue = Energy Efficient Route (RL) | Red = Shortest Path")
plt.savefig("ev_route_result.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# LEARNING CURVE GRAPH
# ------------------------------------------------------------
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("RL Training Learning Curve")
plt.savefig("learning_curve.png", dpi=300)
plt.show()
