from utils.helper_functions import extract_kwargs
import numpy as np
import networkx as nx

def _centrality_select(env_graph, num_batch, measure, reverse=False, rng=None):
    """Selects num_batch most central nodes (according to specified centrality-measure)."""
    # functions to compute centralities
    centrality_functions = {
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality,
        'pagerank': nx.pagerank
    }

    # check if measure is valid
    measure = measure.lower()
    if measure not in centrality_functions:
        raise ValueError('Invalid centrality measure specified.')
    # compute centrality for each node
    centralities = centrality_functions[measure](env_graph)
    node_centralities = list(centralities.items())
    # shuffle centralities
    (rng if rng else np.random).shuffle(node_centralities)
    # return first num_batch nodes with highest centrality
    selected_nodes = np.array(sorted(node_centralities, key=lambda t: t[1], reverse=(not reverse))[:num_batch])[:,0].astype(int)

    return selected_nodes

def degree_centrality_select(**kwargs):
    """Selects num_batch most central (according to degree centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to degree centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'degree', rng=rng)
    return selected_nodes

def degree_centrality_reverse_select(**kwargs):
    """Selects num_batch least central (according to degree centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to degree centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'degree', reverse=True, rng=rng)
    return selected_nodes

def betweenness_centrality_select(**kwargs):
    """Selects num_batch most central (according to betweenness centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to betweenness centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'betweenness', rng=rng)
    return selected_nodes

def betweenness_centrality_reverse_select(**kwargs):
    """Selects num_batch least central (according to betweenness centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to betweenness centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'betweenness', reverse=True, rng=rng)
    return selected_nodes

def closeness_centrality_select(**kwargs):
    """Selects num_batch most central (according to closeness centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to closeness centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'closeness', rng=rng)
    return selected_nodes

def closeness_centrality_reverse_select(**kwargs):
    """Selects num_batch least central (according to closeness centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to closeness centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'closeness', reverse=True, rng=rng)
    return selected_nodes

def eigenvector_centrality_select(**kwargs):
    """Selects num_batch most central (according to eigenvector centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to eigenvector centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'eigenvector', rng=rng)
    return selected_nodes

def eigenvector_centrality_reverse_select(**kwargs):
    """Selects num_batch least central (according to eigenvector centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to eigenvector centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'eigenvector', reverse=True, rng=rng)
    return selected_nodes

def pagerank_centrality_select(**kwargs):
    """Selects num_batch most central (according to pagerank centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to pagerank centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'pagerank', rng=rng)
    return selected_nodes

def pagerank_centrality_reverse_select(**kwargs):
    """Selects num_batch least central (according to pagerank centrality) nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph and num_batch
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # select nodes according to pagerank centrality
    selected_nodes = _centrality_select(env_graph, num_batch, 'pagerank', reverse=True, rng=rng)
    return selected_nodes