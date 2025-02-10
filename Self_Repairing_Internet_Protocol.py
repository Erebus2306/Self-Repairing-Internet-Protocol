import networkx as nx
import glob
import matplotlib.pyplot as plt
import random
import os

# Load all topology files
topology_files = glob.glob(os.path.join("Topology/*.graphml"))  # Change path as needed

# Read and store graphs
network_topologies = []
for file in topology_files:
    G = nx.read_graphml(file)
    G = nx.MultiGraph(G)  # Convert to MultiGraph (ensures multiple edges are supported)
    network_topologies.append(G)

print(f"Loaded {len(network_topologies)} topologies.")

# Preprocessing function for MultiGraph
def preprocess_topology(G):
    # Convert node labels to integers
    G = nx.convert_node_labels_to_integers(G)

    # Add weights if missing for MultiGraph
    for u, v, k in G.edges(keys=True):  # MultiGraph requires "keys=True"
        if "weight" not in G.edges[u, v, k]:
            G.edges[u, v, k]["weight"] = random.randint(1, 10)  # Assign random weight

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    return G

# Apply preprocessing to all MultiGraph topologies
processed_topologies = [preprocess_topology(G) for G in network_topologies]

print(f"Preprocessed {len(processed_topologies)} topologies.")

# Save cleaned MultiGraph topologies
for i, G in enumerate(processed_topologies):
    nx.write_graphml(G, f"cleaned_topology_{i}.graphml")

print("Cleaned MultiGraph topologies saved.")

# Function to visualize a MultiGraph
def visualize_topology(G, title="Network Topology"):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Use spring layout for visualization
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)

    # Handle edge labels for MultiGraph
    edge_labels = {(u, v): G[u][v][list(G[u][v].keys())[0]]["weight"] for u, v in G.edges()}  # Pick one edge per node pair
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.show()

# Visualize the first MultiGraph topology
visualize_topology(processed_topologies[0], title="Sample Cleaned MultiGraph Topology")

# Function to check connectivity of MultiGraph
def check_connectivity(G):
    if nx.is_connected(G):  # MultiGraph must be fully connected
        print("✅ The MultiGraph is fully connected.")
    else:
        print("⚠️ The MultiGraph has disconnected components.")

# Check connectivity for the first topology
check_connectivity(processed_topologies[0])

# Function to find the shortest path in MultiGraph
def test_shortest_path(G):
    try:
        source, target = list(G.nodes())[0], list(G.nodes())[-1]
        path = nx.shortest_path(G, source, target, weight='weight')
        print(f"✅ Shortest path from {source} to {target}: {path}")
    except nx.NetworkXNoPath:
        print("⚠️ No path found between source and target.")

# Test shortest path on first MultiGraph topology
test_shortest_path(processed_topologies[0])