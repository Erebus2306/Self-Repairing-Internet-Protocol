import time
import numpy as np
import threading
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler
import networkx as nx
import glob
import os
import random

app = dash.Dash(__name__)

# Load and preprocess topologies
def load_topologies():
    try:
        topology_files = glob.glob(os.path.join("Topology/*.graphml"))
        if not topology_files:
            raise FileNotFoundError("No GraphML files found in the Topology directory.")
        network_topologies = [nx.read_graphml(file) for file in topology_files]
        return [preprocess_topology(G) for G in network_topologies]
    except Exception as e:
        print(f"Error loading topologies: {e}")
        return []

def preprocess_topology(G):
    G = nx.convert_node_labels_to_integers(nx.MultiGraph(G))
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k].setdefault("weight", random.randint(1, 10))
        G.edges[u, v, k].setdefault("failure", False)
        G.edges[u, v, k].setdefault("congestion", random.uniform(0, 1))
    return G

# Global variables to keep track of monitoring state and historical failure data
monitoring_active = False
historical_failures = []
processed_topologies = []

def simulate_link_failures(G):
    for u, v, k in G.edges(keys=True):
        if random.random() < 0.1:  # 10% chance of failure
            G.edges[u, v, k]["failure"] = True
            historical_failures.append((u, v, k, time.time()))
        else:
            G.edges[u, v, k]["failure"] = False

def simulate_congestion_levels(G):
    for u, v, k in G.edges(keys=True):
        G.edges[u, v, k]["congestion"] = random.uniform(0, 1)

def dynamic_qos_routing(G, source, target):
    try:
        # Check if the target is reachable from the source
        if not nx.has_path(G, source, target):
            raise ValueError(f"Target {target} cannot be reached from given sources")

        # Find all shortest paths considering congestion and failures
        paths = nx.all_shortest_paths(G, source=source, target=target, weight='weight')
        for path in paths:
            if all(not G.edges[u, v, k]['failure'] and G.edges[u, v, k]['congestion'] < 0.8 for u, v, k in zip(path[:-1], path[1:], range(len(path)-1))):
                return path
        raise ValueError("No QoS-aware path available")
    except Exception as e:
        print(f"❌ {e}")
        return None

def update_topology(G):
    # Implement the topology update logic here
    pass

def monitor_network(G, interval=5):
    global monitoring_active
    while monitoring_active:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔍 {timestamp}: Monitoring network...")
        simulate_link_failures(G)
        simulate_congestion_levels(G)
        path = dynamic_qos_routing(G, 0, max(G.nodes()))
        if path:
            print(f"✅ QoS-aware path found: {path}")
        else:
            print("❌ No QoS-aware path available.")
        update_topology(G)
        time.sleep(interval)
    print(f"⏹️ {timestamp}: Monitoring stopped.")

@app.callback(
    Output('interval-component', 'disabled'),
    [Input('start-button', 'n_clicks'), Input('stop-button', 'n_clicks')]
)
def toggle_monitoring(start_clicks, stop_clicks):
    global monitoring_active
    ctx = dash.callback_context

    if not ctx.triggered:
        return True
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-button':
        if not monitoring_active:
            monitoring_active = True
            threading.Thread(target=monitor_network, args=(processed_topologies[0],)).start()
        return False
    elif button_id == 'stop-button':
        monitoring_active = False
        return True

@app.callback(
    Output('cytoscape-network', 'elements'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n_intervals):
    return generate_cytoscape_graph(processed_topologies[0])

# Visualization Dashboard
def generate_cytoscape_graph(G):
    elements = [{"data": {"id": str(node), "label": str(node)}} for node in G.nodes()]
    elements += [{"data": {"source": str(u), "target": str(v), "weight": G.edges[u, v, k]["weight"], "failure": G.edges[u, v, k]["failure"], "congestion": G.edges[u, v, k]["congestion"]}} for u, v, k in G.edges(keys=True)]
    return elements

if __name__ == "__main__":
    processed_topologies = load_topologies()
    if processed_topologies:
        app.layout = html.Div([
            html.H1("Network Monitoring Dashboard"),
            html.Div([
                html.Button("Start Monitoring", id="start-button", n_clicks=0, style={'margin-right': '10px'}),
                html.Button("Stop Monitoring", id="stop-button", n_clicks=0)
            ], style={'margin-bottom': '20px'}),
            cyto.Cytoscape(
                id='cytoscape-network',
                elements=generate_cytoscape_graph(processed_topologies[0]),
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '600px'}
            ),
            dcc.Interval(id='interval-component', interval=5000, n_intervals=0, disabled=True)
        ])
        app.run_server(debug=True, use_reloader=False)
    else:
        print("No topologies loaded. Exiting.")