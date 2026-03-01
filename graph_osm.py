import osmnx as ox
import matplotlib.pyplot as plt

# Load local OSM file
G = ox.graph_from_xml("map1.osm", simplify=True)

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

# Plot
ox.plot_graph(G)