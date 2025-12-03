import networkx as nx
import numpy as np


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
