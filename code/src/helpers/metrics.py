import networkx as nx
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


def get_node_count(G: nx.Graph):
    return G.number_of_nodes()


def get_edge_count(G: nx.Graph):
    return G.number_of_edges()


def get_density(G: nx.Graph):
    return nx.density(G)


def get_diameter(G: nx.Graph):
    return nx.diameter(G)


def get_largest_connected_component(G: nx.Graph):
    nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(nodes)


def get_average_shortest_path(G: nx.Graph):
    if nx.is_connected(G):
        largest_connected_component = G
    else:
        largest_connected_component = get_largest_connected_component(G)
    return nx.average_shortest_path_length(largest_connected_component)


def get_assortativity(G: nx.Graph):
    return float(nx.degree_assortativity_coefficient(G))


def get_node_degrees_distribution(G: nx.Graph):
    return [G.degree(n) for n in G.nodes]


def get_connected_component_size_distribution(G: nx.Graph):
    return [len(c) for c in nx.connected_components(G)]


def get_clustering_coefficient_distribution(G: nx.Graph):
    return list(nx.clustering(G).values())


def get_betweenness_centrality_distribution(G: nx.Graph):
    betweenness = nx.centrality.betweenness_centrality(G)
    return list(betweenness.values())


def get_community_count(partition: list[set]):
    return len(partition)


def get_modularity(G: nx.Graph, partition: list[set]):
    if len(partition) > 0:
        return nx.community.quality.modularity(G, partition)
    else:
        return -1


def get_mean_internal_edge_ratio(G: nx.Graph, partition: list[set]):
    ratios = []
    for community in partition:
        node_list = set(community)
        internal_edges = 0
        external_edges = 0
        for u in node_list:
            for v in G.neighbors(u):
                if v in node_list:
                    internal_edges += 1
                else:
                    external_edges += 1

        # internal edges were counted twice (u->v and v->u)
        internal_edges //= 2

        ratio = internal_edges / external_edges if external_edges > 0 else 1
        ratios.append(ratio)

    return sum(ratios) / len(ratios) if ratios else 0


def get_mean_jaccard_community_similarity(reference: dict, current_data: dict):
    """
    Compute average Jaccard similarity of community affiliations
    for users present in both current and reference networks.

    Args:
        reference (dict): Dictionary linking users to community IDs
            according to their affiliations in the reference network.
        current_data (dict): Dictionary linking users to community IDs
            according to their affiliations in the current network.

    Returns:
        float: Mean Jaccard similarity across common users (0..1).
    """
    if not reference or not current_data:
        return 0

    common_users = set(reference.keys()) & set(current_data.keys())
    if len(common_users) == 0:
        return 0

    similarities = []

    for user in common_users:
        ref_affiliations = set(reference[user])
        cur_affiliations = set(current_data[user])

        if ref_affiliations or cur_affiliations:
            intersection = ref_affiliations & cur_affiliations
            union = ref_affiliations | cur_affiliations
            jaccard = len(intersection) / len(union)
            similarities.append(jaccard)

    return np.mean(similarities) if similarities else 0


def get_community_entry_counts(reference: dict, current_data: dict):
    """
    Compute number of new users in a community by comparison with
    a reference network.

    Args:
        reference (dict): Dictionary linking users to community IDs
            according to their affiliations in the reference network.
        current_data (dict): Dictionary linking users to community IDs
            according to their affiliations in the current network.

    Returns:
        int: Number of users present in the current data that were not
            in the reference data.
    """
    if not reference or not current_data:
        return 0

    new_users = set(current_data.keys()) - set(reference.keys())
    return len(new_users)


def get_community_exit_counts(reference: dict, current_data: dict):
    """
    Compute number of users who left a community by comparison with
    a reference network.

    Args:
        reference (dict): Dictionary linking users to community IDs
            according to their affiliations in the reference network.
        current_data (dict): Dictionary linking users to community IDs
            according to their affiliations in the current network.

    Returns:
        int: Number of users present in the current data that were not
            in the reference data.
    """
    if not reference or not current_data:
        return 0

    former_users = set(reference.keys()) - set(current_data.keys())
    return len(former_users)


def perform_ks_test(reference: list, current_data: list):
    """
    Perform a Kolmogorov-Smirnov test to check whether two samples
    are statistically likely to follow the same distribution.

    Args:
        reference (list): First data sample.
        current_data (list): Second data sample.

    Returns:
        dict: Dictionary containing the KS test statistic and p-value
            with the format {"statistic": float, "p_value": float}.
    """
    x = np.asarray(reference)
    y = np.asarray(current_data)

    ks_stat, ks_pval = ks_2samp(x, y)
    ks_result = {"statistic": ks_stat, "p_value": ks_pval}

    return ks_result


def compute_js_distance(reference: list, current_data: list, num_bins: int = 50):
    """
    Compute the Jensen-Shannon distance between two samples.

    Args:
        reference (list): First data sample.
        current_data (list): Second data sample.
        num_bins (int, optional): Number of bins used to build
            histograms for estimating probability distributions
            for the JS distance computation (default: 50).

    Returns:
        float: JS distance between the two empirical distributions.
    """
    x = np.asarray(reference)
    y = np.asarray(current_data)

    # Create a unified bin range
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Compute normalized histograms (empirical probability distributions)
    p_hist, _ = np.histogram(x, bins=bins, density=True)
    q_hist, _ = np.histogram(y, bins=bins, density=True)

    # Add small epsilon to avoid zero-probability issues
    eps = 1e-12
    p = p_hist / (p_hist + eps).sum()
    q = q_hist / (q_hist + eps).sum()

    js_distance = jensenshannon(p, q)

    return js_distance
