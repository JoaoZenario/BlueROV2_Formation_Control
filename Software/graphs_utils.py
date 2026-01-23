import numpy as np
from typing import List, Tuple, Dict, Any

from scipy.sparse.csgraph import connected_components

import networkx as nx
import matplotlib.pyplot as plt


def get_neighbor_lists_from_adjacency(A: np.ndarray) -> Dict[int, List[int]]:
    """
    Converts an adjacency matrix A (0-based indices) into a dictionary
    where keys are agent IDs and values are their neighbor IDs

    Args:
        A (np.ndarray): NxN adjacency matrix (0-1). Assumed undirected.

    Returns:
        Dict[int, List[int]]: Dictionary mapping 0-based agent ID (0 to N-1)
                              to a list of 0-based neighbor IDs.
    """
    N = A.shape[0]
    neighbors_dict: Dict[int, List[int]] = {}

    # Iterate through each row of the adjacency matrix (i is 0-based)
    for i in range(N):
        agent_id = i  # 0-based ID for the current agent (KEY)

        # Use numpy.where to find the indices j where A[i, j] is 1
        # neighbor_indices is already 0-based
        neighbor_indices = np.where(A[i, :] == 1)[0]

        # Convert the NumPy array of indices to a standard list
        neighbors = neighbor_indices.tolist()

        # Store in the dictionary
        neighbors_dict[agent_id] = neighbors

    return neighbors_dict


def neighbors_to_und_edges(neighbors_dict: Dict[int, List[int]]) -> np.ndarray:
    """
    Converts a neighbor list dictionary to an undirected edge list.
    Args:
        neighbors_dict (Dict[int, List[int]]): Dictionary mapping 0-based agent ID
                                                to a list of 0-based neighbor IDs.
    Returns:
        np.ndarray: Matrix of edges, where each row is [i, j] with i < j.
    """
    edges_set = set()

    for i, neighbors in neighbors_dict.items():
        for j in neighbors:
            # To avoid duplicates, always store the edge with the smaller ID first
            if i < j:
                edges_set.add((i, j))
            else:
                edges_set.add((j, i))

    # Convert the set of edges to a NumPy array
    edges_array = np.array(list(edges_set))

    return edges_array


def adj_to_und_edges(A: np.ndarray) -> np.ndarray:
    """
    Converts an adjacency matrix A to a list of edges E.
    The edges are listed as (i, j) where i < j (undirected).

    Args:
        N (int): Number of vertices.
        A (np.ndarray): NxN adjacency matrix.

    Returns:
        np.ndarray: Matrix of edges, where each row is [i, j].
    """
    # Find indices where A[i, j] is True (i.e., edge exists)
    rows, cols = np.where(A)

    # Combine them into pairs (i, j)
    E_full = np.stack([rows, cols], axis=1)

    # Filter for i < j to list each undirected edge only once
    mask = E_full[:, 0] < E_full[:, 1]
    E = E_full[mask]

    # Add 1 to indices if you want 1-based indexing (like MATLAB)
    for i in range(E.shape[0]):
        E[i, 0] += 1
        E[i, 1] += 1

    # If the rest of your code uses 0-based indexing (standard Python), skip this.
    # We will return 0-based indices here:
    return E


def is_connected_scipy(A: np.ndarray) -> bool:
    """
    Checks if a graph with adjacency matrix A is connected using scipy.

    Args:
        A (np.ndarray): NxN adjacency matrix (undirected).

    Returns:
        bool: True if connected, False otherwise.
    """
    # connected_components returns the number of components and component labels
    n_components, _ = connected_components(A, directed=False, return_labels=True)
    return n_components == 1


def generate_random_connected_graph(
    N: int, d: int, d_max: float, d_min: float, l: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generates a random proximity graph embedded in (R^d)^N.

    Args:
        N (int): Number of vertices/agents.
        d (int): Dimension of the state (e.g., 2 for a 2D plane).
        d_max (float): Maximum distance for two nodes to be neighbors (radius).
        d_min (float): Minimum distance for two nodes to be neighbors.
        l (float): Length of the side of the square/hypercube [0, l]^d.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        - P (np.ndarray): NxD matrix of positions.
        - A (np.ndarray): NxN adjacency matrix (0-1).
        - E (np.ndarray): Edge set, matrix of size NE x 2 (0-based indices).
        - NE (int): Number of edges.
    """
    if d_min >= d_max:
        raise ValueError("d_min must be strictly less than d_max.")

    connected = False

    # Pre-allocate II for vectorization
    II = np.ones((N, 1))

    while not connected:
        # 1. Generate random positions on the hypercube [0, l]^d
        # P is NxD
        P = l * np.random.rand(N, d)

        # 2. Compute squared distances between all nodes
        D2 = np.zeros((N, N))

        # Vectorized distance computation
        for dd in range(d):
            # Extract the dd-th column/dimension
            X_dd = P[:, dd].reshape(-1, 1)  # Column vector (N x 1)

            # This computes (X_i - X_j)^T
            # The operation is (P_i - P_j) for the dd-th component
            X = X_dd @ II.T - II @ X_dd.T
            D2 += X**2

        # 3. Compute distances and Adjacency Matrix
        dist = np.sqrt(D2)

        # Two nodes are neighbors if d_min <= dist <= d_max
        A = (dist <= d_max) & (dist >= d_min)
        A = A.astype(int) # Convert boolean to integer (0 or 1)
        A = np.tril(A, k=-1) # In order to transform in a directed system (strictily lower triangular matrix)

        for i in range(1, N):
            # If the sum of row i is 0, this agent follows no one.
            if np.sum(A[i, :]) == 0:
                # Connect to any random agent with a lower index.
                random_parent = np.random.randint(0, i)
                A[i, random_parent] = 1

        # 4. Check Connectivity
        # MATLAB: connected = all( all( A^N ) ) is an algebraic check for connectivity
        # which can be slow and less reliable than graph-based methods.
        # Python uses a robust graph-based check (Breadth-First Search / Union-Find).
        connected = is_connected_scipy(A)

    # 5. Final Processing
    # Remove self-loops (already handled by dist check, but good practice)
    np.fill_diagonal(A, 0)

    # Get edge set E and number of edges NE
    E = adj_to_und_edges(A)
    NE = E.shape[0]

    # Return P, A, E (0-based indexing), NE
    return P, A, E, NE, dist


def plot_graph_from_adjacency(
    P: np.ndarray,
    A: np.ndarray,
    title: str = "Connected Proximity Graph Topology",
    show: bool = True,
    **kwargs: Dict[str, Any],
):
    """
    Plots a graph given its positions and adjacency matrix.

    Args:
        P (np.ndarray): NxD matrix of node positions.
        A (np.ndarray): NxN adjacency matrix (0-1).
        title (str): Title for the plot.
        **kwargs: Optional keyword arguments passed to nx.draw_networkx,
                  like node_color, node_size, etc.
    """
    N = P.shape[0]

    # 1. Create the NetworkX Graph object
    G = nx.DiGraph(A)

    # 2. Map the positions from the P matrix to the required NetworkX dictionary format
    # The dictionary keys must be the node indices (0 to N-1)
    # The values must be the coordinates (e.g., [x, y]) - use only first 2 dimensions for visualization
    pos = {i: P[i, :2] for i in range(N)}

    # 3. Prepare the labels (Agent 1, Agent 2, ...)
    labels = {i: str(i + 1) for i in range(N)}  # Use 1-based indexing for display

    # 4. Set default plotting parameters, allowing kwargs to override
    draw_params = {
        "node_color": "#007ACC",
        "node_size": 400,
        "edge_color": "#AAAAAA",
        "width": 1.5,
        "arrows": True,
        "arrowstyle": '-|>',
        "arrowsize": 20,
        "with_labels": False,  # We'll draw labels separately for better control
        "font_weight": "bold",
        "font_color": "white",
        **kwargs,
    }

    # 5. Plotting
    plt.figure(figsize=(8, 8))
    plt.title(title)

    # Draw nodes and edges
    nx.draw(G, pos, **draw_params)

    # Draw labels separately
    nx.draw_networkx_labels(G, pos, labels=labels, font_color="white", font_size=10)

    # Ensure the axes match the bounds of the generated positions
    x_min, x_max = np.min(P[:, 0]), np.max(P[:, 0])
    y_min, y_max = np.min(P[:, 1]), np.max(P[:, 1])
    plt.xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
    plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.6)
    if show:
        plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    # Set parameters
    N = 10  # 10 vertices
    d = 2  # 2D plane
    d_max = 0.4  # Connection radius
    d_min = 0.0  # No minimum distance requirement
    l = 1.0  # Square side length (unit square)

    P, A, E, NE, dist = generate_random_connected_graph(N, d, d_max, d_min, l)

    print(f"--- Generated Connected Proximity Graph (N={N}, D={d}) ---")
    print(f"Total Edges (NE): {NE}")
    print(f"Adjacency Matrix (A): \n", A)
    print(f"Positions (P):\n", P)
    print(f"Edges (E): \n", E)

    if NE > 0:
        print(f"\nConnectivity Check: {is_connected_scipy(A)}")

    plot_graph_from_adjacency(P, A)
