# EV Energy Efficient Route using Reinforcement Learning 🚗🔋

This project implements a Reinforcement Learning based navigation system that finds the most energy-efficient route for an Electric Vehicle (EV).

## Idea

Traditional navigation systems minimize **distance**.
This project minimizes **battery energy consumption** considering:

* Road distance
* Traffic conditions
* Road slope

The RL agent learns the optimal route through interaction with the environment.

## Algorithms Used

* Q-Learning (Reinforcement Learning)
* Dijkstra Algorithm (baseline comparison)

## Features

* Uses real OpenStreetMap road network
* Interactive start & destination selection
* Energy-aware route optimization
* Comparison with shortest path
* Learning curve visualization

## Output

Blue Path → Energy Efficient Route (RL)
Red Path → Shortest Path (Dijkstra)

## How to Run

```bash
pip install -r requirements.txt
python test1.py
```

Then click start and destination on the map.

## Results

The RL agent successfully learns routes that consume less battery energy compared to the shortest path.

## Author

Logeshwaran
