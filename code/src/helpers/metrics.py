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


def get_average_shortest_path(G: nx.Graph):
    if nx.is_connected(G):
        largest_connected_component = G
    else:
        nodes = max(nx.connected_components(G), key=len)
        largest_connected_component = G.subgraph(nodes)
    return nx.average_shortest_path_length(largest_connected_component)


def get_assortativity(G: nx.Graph):
    return float(nx.degree_assortativity_coefficient(G))


def get_connected_component_size_distribution(G: nx.Graph):
    return [len(c) for c in nx.connected_components(G)]


def get_node_degrees_distribution(G: nx.Graph):
    return [G.degree(n) for n in G.nodes]


def get_clustering_coefficient_distribution(G: nx.Graph):
    return list(nx.clustering(G).values())


def get_betweenness_centrality_distribution(G: nx.Graph):
    betweenness = nx.centrality.betweenness_centrality(G)
    return list(betweenness.values())


def get_community_count(partition: list[set]):
    return len(partition)


def get_modularity(G: nx.Graph, partition: list[set]):
    return nx.community.quality.modularity(G, partition)
