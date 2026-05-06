import numpy as np
from typing import List, Tuple, Dict, Any

from scipy.sparse.csgraph import connected_components

import networkx as nx
import matplotlib.pyplot as plt


def compute_directed_edge_list(A: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Edge list for a directed graph adjacency matrix.

    Returns an array E where each row is (i, j) such that A[i, j] == 1,
    and NE = number of directed edges.
    """
    A = np.asarray(A)
    edges = []
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1:
                edges.append((i, j))

    E = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
    return E, len(edges)


def is_rooted_directed_tree(A: np.ndarray, root: int = 0) -> bool:
    """
    Check whether adjacency A defines a rooted directed spanning tree
    under convention A[i, j] = 1 meaning i has parent j.
    """
    A = np.asarray(A)
    N = A.shape[0]

    if np.sum(A[root]) != 0:
        return False

    for i in range(N):
        if i == root:
            continue
        if np.sum(A[i]) != 1:
            return False

    for i in range(N):
        visited = set()
        current = i
        while current != root:
            parents = np.where(A[current] == 1)[0]
            if len(parents) != 1:
                return False
            parent = int(parents[0])
            if parent in visited:
                return False
            visited.add(parent)
            current = parent

    return True


def build_directed_tree_adjacency(
    N: int,
    tree_type: str = "leader_star",
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Build a rooted directed tree adjacency matrix under convention
    A[i, j] = 1 means i uses information from parent j.
    """
    A = np.zeros((N, N), dtype=int)

    if N <= 1:
        return A

    if tree_type == "leader_star":
        for i in range(1, N):
            A[i, 0] = 1
    elif tree_type == "leader_chain":
        for i in range(1, N):
            A[i, i - 1] = 1
    elif tree_type == "random_tree":
        rng = np.random.default_rng(random_seed)
        for i in range(1, N):
            parent = int(rng.integers(0, i))
            A[i, parent] = 1
    else:
        raise ValueError(f"Unknown tree_type '{tree_type}'")

    return A


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
    N: int,
    d: int,
    d_max: float,
    d_min: float,
    l: float,
    random_seed: int | None = None,
    max_tries: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Generate a random graph embedded in [0,l]^d.

    Parameters
    ----------
    N : int
        Number of agents.
    d : int
        State dimension (2 or 3).
    d_max : float
        Connectivity radius.
    d_min : float
        Collision-avoidance threshold.
    l : float
        Workspace side length.
    mode : bool
        Graph mode:
            False -> undirected
            True  -> directed
    fov : bool
        If True and mode is directed, each follower is forced to keep at most one parent.
    random_seed : int | None
        Random seed.
    max_tries : int
        Maximum number of rejection-sampling attempts.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]
        P   : positions, shape (N,d)
        A   : adjacency matrix, shape (N,N)
        E   : edge list, shape (NE,2)
        NE  : number of edges
        dist: pairwise distance matrix, shape (N,N)
    """
    if N < 1:
        raise ValueError("N must be positive.")
    if d not in (2, 3):
        raise ValueError("d must be 2 or 3.")
    if d_min >= d_max:
        raise ValueError("d_min must be strictly less than d_max.")
    if l <= 0:
        raise ValueError("l must be positive.")
    if max_tries < 1:
        raise ValueError("max_tries must be at least 1.")

    rng = np.random.default_rng(random_seed)

    II = np.ones((N, 1))

    for _ in range(max_tries):
        # 1) Random positions
        P = l * rng.random((N, d))

        # 2) Pairwise squared distances
        D2 = np.zeros((N, N), dtype=float)
        for dd in range(d):
            X_dd = P[:, dd].reshape(-1, 1)
            X = X_dd @ II.T - II @ X_dd.T
            D2 += X**2

        dist = np.sqrt(D2)

        # 3) Base proximity graph: neighbors if d_min <= dist <= d_max
        base = ((dist <= d_max) & (dist >= d_min)).astype(int)
        np.fill_diagonal(base, 0)

        # =====================================================
        # Undirected case
        # =====================================================
        A = np.maximum(base, base.T)
        np.fill_diagonal(A, 0)

        if is_connected_scipy(A):
            E = adj_to_und_edges(A)
            NE = E.shape[0]

            consensus = check_consensus_capability(A)
            if consensus:
                print("Condition Met: Undirected connected graph generated.")
            else:
                print("Condition Failed.")

            return P, A, E, NE, dist


def plot_graph_from_adjacency(
    positions: np.ndarray,
    adjacency: np.ndarray,
    directed: bool = True,
    title: str = "Graph Topology",
    show: bool = True,
    save_path: str | None = None,
    ax=None,
):
    """
    Plot a graph from an adjacency matrix.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2) or (N, 3)
        Node positions.
    adjacency : np.ndarray, shape (N, N)
        Adjacency matrix.
    directed : bool
        Whether to interpret the graph as directed.
    title : str
        Figure title.
    show : bool
        Whether to call plt.show().
    save_path : str | None
        If provided, save the figure to this path.
    ax : matplotlib axis | None
        Optional existing axis.
    """
    positions = np.asarray(positions, dtype=float)
    adjacency = np.asarray(adjacency)

    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError("positions must have shape (N,2) or (N,3)")
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if adjacency.shape[0] != positions.shape[0]:
        raise ValueError("adjacency and positions size mismatch")

    dim = positions.shape[1]
    created_fig = False

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        if dim == 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.figure

    # plot nodes
    if dim == 3:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=50)
        for i, p in enumerate(positions):
            ax.text(p[0], p[1], p[2], f"{i}", fontsize=9)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], s=50)
        for i, p in enumerate(positions):
            ax.text(p[0], p[1], f"{i}", fontsize=9)

    # plot edges
    N = positions.shape[0]
    for i in range(N):
        for j in range(N):
            if adjacency[i, j] != 1:
                continue

            pi = positions[i]
            pj = positions[j]

            if dim == 3:
                ax.plot(
                    [pi[0], pj[0]],
                    [pi[1], pj[1]],
                    [pi[2], pj[2]],
                    color="gray",
                    alpha=0.8,
                )
            else:
                ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color="gray", alpha=0.8)

            if directed and dim == 2:
                d = pj[:2] - pi[:2]
                n = np.linalg.norm(d)
                if n > 1e-9:
                    d = d / n
                    start = pi[:2] + 0.1 * d
                    delta = (pj[:2] - pi[:2]) - 0.2 * d
                    ax.arrow(
                        start[0],
                        start[1],
                        delta[0],
                        delta[1],
                        head_width=0.08,
                        head_length=0.12,
                        fc="gray",
                        ec="gray",
                        length_includes_head=True,
                        alpha=0.7,
                    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if dim == 3:
        ax.set_zlabel("Z")

    mins = np.min(positions, axis=0)
    maxs = np.max(positions, axis=0)
    pad = 0.5
    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    if dim == 3:
        ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show and created_fig:
        plt.show()

    return fig, ax


# * Function to verifiy if there exists a spanning tree in the generated graph.
def check_consensus_capability(A):
    # For a directed graph, all non-zero eigenvalues of L have positive real parts and 0 is a simple eigenvalue if and only if the directed graph contains a directed spanning tree.

    # Degree Matrix.
    # D is a diagonal matrix where each entry is the sum of each matrix A row (d_ii = ∑_j a_ij).
    D = np.diag(np.sum(A, axis=1))
    # Laplacian Matrix.
    # The Laplacian matrix is the difference between the Degree matrix and the Adjacency matrix (L = D - A, from equation 2.15).
    L = D - A
    # Compute Eigenvalues.
    eigenvalues = np.linalg.eigvals(L)
    # Count zeros (using a small epsilon for numerical stability).
    zeros = [ev for ev in eigenvalues if abs(ev) < 1e-10]
    non_zeros = [ev for ev in eigenvalues if abs(ev) >= 1e-10]

    # Verify if it's a simple zero.
    is_simple_zero = len(zeros) == 1

    # Positive Real Parts.
    # .real is used because eigenvalues can be complex numbers (a + bi).
    all_positive_real = all(ev.real > 0 for ev in non_zeros)

    if is_simple_zero and all_positive_real:
        return True
    else:
        return False


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
