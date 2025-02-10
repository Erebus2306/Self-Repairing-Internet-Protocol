import networkx as nx
import glob
import matplotlib.pyplot as plt
import random
import os
import time
import numpy as np
import threading
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler

# Load and preprocess topologies
def load_topologies():
    topology_files = glob.glob(os.path.join("Topology/*.graphml"))
    network_topologies = [nx.read_graphml(file) for file in topology_files]
    return [preprocess_topology(G) for G in network_topologies]

def preprocess_topology(G):
    G = nx.convert_node_labels_to_integers(nx.MultiGraph(G))
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k].setdefault("weight", random.randint(1, 10))
        G.edges[u, v, k].setdefault("latency", random.uniform(1, 50))
        G.edges[u, v, k].setdefault("bandwidth", random.randint(1, 100))
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

processed_topologies = load_topologies()
print(f"Preprocessed {len(processed_topologies)} topologies.")

# Enhanced BGP-like Topology Updates with Route Preferences and AS-like Behavior
def update_topology(G):
    if random.random() < 0.3:
        node_to_remove = random.choice(list(G.nodes()))
        G.remove_node(node_to_remove)
        print(f"🔄 BGP Update: Removed node {node_to_remove}")
    
    if random.random() < 0.3:
        new_node = max(G.nodes()) + 1
        potential_neighbors = list(G.nodes())
        random.shuffle(potential_neighbors)
        for neighbor in potential_neighbors[:2]:
            G.add_edge(new_node, neighbor, weight=random.randint(1, 10), latency=random.uniform(1, 50), bandwidth=random.randint(1, 100))
        print(f"🔄 BGP Update: Added node {new_node} with links to {potential_neighbors[:2]}")
    
    for node in list(G.nodes()):
        if random.random() < 0.2:
            neighbors = list(G.neighbors(node))
            if neighbors:
                withdrawn_neighbor = random.choice(neighbors)
                G.remove_edge(node, withdrawn_neighbor)
                print(f"📉 BGP Route Withdrawal: Node {node} lost link to {withdrawn_neighbor}")

def dynamic_qos_routing(G, source, target):
    try:
        path = nx.shortest_path(G, source, target, weight="latency")
        print(f"⚡ QoS-aware path selected: {path}")
        return path
    except nx.NetworkXNoPath:
        print("❌ No QoS-aware path available.")
        return None

# Multi-Agent Q-Learning for Failure Prediction
class QLearningFailurePredictor:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max([self.get_q_value(next_state, a) for a in ["reroute", "reinforce", "do_nothing"]], default=0)
        self.q_table[(state, action)] = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * (reward + self.gamma * best_next_action)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["reroute", "reinforce", "do_nothing"])
        return max(["reroute", "reinforce", "do_nothing"], key=lambda a: self.get_q_value(state, a))

    def train(self, G):
        for _ in range(1000):
            node = random.choice(list(G.nodes()))
            action = self.select_action(node)
            reward = self.simulate_action(G, node, action)
            next_state = node
            self.update_q_value(node, action, reward, next_state)
        print("🧠 Q-learning training complete.")

    def simulate_action(self, G, node, action):
        if action == "reroute":
            return 10 if len(list(G.neighbors(node))) > 1 else -10
        elif action == "reinforce":
            return 5 if random.random() > 0.5 else -5
        return -1

q_predictor = QLearningFailurePredictor()
q_predictor.train(processed_topologies[0])

# Monitoring and Rerouting
monitoring_active = False

def monitor_network(G, interval=5):
    global monitoring_active
    while monitoring_active:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔍 {timestamp}: Monitoring network...")
        dynamic_qos_routing(G, 0, max(G.nodes()))
        update_topology(G)
        time.sleep(interval)
    print(f"⏹️ {timestamp}: Monitoring stopped.")

# Visualization Dashboard
app = dash.Dash(__name__)

def generate_cytoscape_graph(G):
    elements = [{"data": {"id": str(node), "label": str(node)}} for node in G.nodes()]
    elements += [{"data": {"source": str(u), "target": str(v), "weight": G.edges[u, v, k]["weight"]}} for u, v, k in G.edges(keys=True)]
    return elements

app.layout = html.Div([
    html.H1("Network Monitoring Dashboard"),
    html.Button("Start Monitoring", id="start-button", n_clicks=0),
    html.Button("Stop Monitoring", id="stop-button", n_clicks=0),
    cyto.Cytoscape(
        id='cytoscape-network',
        elements=generate_cytoscape_graph(processed_topologies[0]),
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '600px'}
    ),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)