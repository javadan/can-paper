import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_topology(topology_type, n_neurons=200, n_digits=10):
    """
    Simulates connectivity.
    topology_type: 'Snake' or 'Ring'
    """
    neurons_per_digit = n_neurons // n_digits
    weights = np.zeros((n_neurons, n_neurons))
    
    # Assign preferred digits (0,0,... 1,1,... etc)
    neuron_preferences = np.repeat(np.arange(n_digits), neurons_per_digit)
    
    print(f"Generating {topology_type} Topology...")

    for i in range(n_neurons):
        for j in range(n_neurons):
            digit_i = neuron_preferences[i]
            digit_j = neuron_preferences[j]
            
            # Distance logic
            raw_dist = abs(digit_i - digit_j)
            
            if topology_type == 'Ring':
                # Modulo distance (wrap-around)
                # Distance between 0 and 9 is 1
                dist = min(raw_dist, n_digits - raw_dist)
            else:
                # Snake (Linear) distance
                # Distance between 0 and 9 is 9
                dist = raw_dist
            
            # Weight logic (Symmetric Attractor Weights)
            if dist == 0:
                # Self/Same Digit
                weights[i, j] = 1.0 + np.random.normal(0, 0.1)
            elif dist == 1:
                # Immediate Neighbor
                weights[i, j] = 0.5 + np.random.normal(0, 0.1)
            else:
                weights[i, j] = 0.0 
                
    return weights, neuron_preferences

def plot_weight_matrix(weights, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(weights, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Connection Strength")
    plt.title(f"{title} - Weight Matrix ($W_{{attr}}$)")
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    
    # Highlight the corners for Ring
    if title == 'Ring':
        # Add visual markers for the wrap-around weights
        plt.text(180, 20, "Wrap-around", color='white', fontsize=8, ha='center', fontweight='bold')
        plt.text(20, 180, "Wrap-around", color='white', fontsize=8, ha='center', fontweight='bold')

    plt.tight_layout()
    filename = f"vis_matrix_{title.lower()}.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

def plot_force_directed_graph(weights, labels, title):
    print(f"Computing layout for {title}...")
    
    G = nx.Graph()
    n_neurons = weights.shape[0]
    
    for i in range(n_neurons):
        G.add_node(i, digit=labels[i])
    
    # Filter edges to keep visualization clean
    threshold = 0.4
    rows, cols = np.where(weights > threshold)
    for r, c in zip(rows, cols):
        if r != c: 
            G.add_edge(r, c, weight=weights[r, c])

    # Spring Layout
    pos = nx.spring_layout(G, seed=42, iterations=150, k=0.15)
    
    plt.figure(figsize=(10, 8))
    
    node_colors = [labels[i] for i in G.nodes()]
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=60, cmap=plt.cm.tab10, node_color=node_colors, alpha=0.9)
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray')
    
    plt.title(f"{title} Topology\n(Force-Directed Layout)")
    plt.axis('off')
    
    # Custom Legend - Moved to the right, outside the plot area
    digits = np.unique(labels)
    cmap = plt.cm.tab10
    norm = plt.Normalize(vmin=0, vmax=9)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(norm(d)), markersize=10, label=f'{d}') for d in digits]
    
    # bbox_to_anchor=(1, 1) places the legend at the top-right corner, outside the axes
    plt.legend(handles=handles, title="Digit", loc='upper left', bbox_to_anchor=(1, 1))
    
    filename = f"vis_graph_{title.lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    for topo in ['Snake', 'Ring']:
        w, l = generate_topology(topo, 200, 10)
        plot_weight_matrix(w, topo)
        plot_force_directed_graph(w, l, topo)
