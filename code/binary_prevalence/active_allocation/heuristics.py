from utils.helper_functions import extract_kwargs
import numpy as np

def random_select(**kwargs):
    """Randomly selects num_batch unobserved nodes."""
    # extract required arguments
    required_keys = ['surrogate', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get observed_infection_states from surrgate and num_batch
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # get unobserved nodes
    unobserved_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == -1]
    # randomly select num_batch nodes from unobserved_nodes
    selected_nodes = rng.choice(unobserved_nodes, min(num_batch, len(unobserved_nodes)), replace=False)

    return selected_nodes

def reactive_select(**kwargs):
    """Randomly selects num_batch unobserved nodes that are connected to newly observed nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'surrogate', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)
    
    # get last_observed_nodes and last_observed_nodes from surrogate, num_batch, and env_graph
    last_observed_nodes = kwargs['surrogate'].get_last_observed_nodes()
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    num_batch = kwargs['num_batch']
    env_graph = kwargs['env_graph']
    rng = kwargs['rng']

    # get neighbors of newly observed nodes
    new_observed_neighbors = list(set([neighbor for node in last_observed_nodes for neighbor in env_graph.neighbors(node)]))
    # get unobserved neighbors of newly observed nodes
    unobserved_new_observed_neighbors = [neighbor for neighbor in new_observed_neighbors if observed_infection_states[neighbor] == -1]
    # randomly select num_batch from unobserved_new_observed_neighbors
    selected_nodes = rng.choice(unobserved_new_observed_neighbors, min(num_batch, len(unobserved_new_observed_neighbors)), replace=False).astype(int)
    # if not enough unobserved_new_observed_neighbors, randomly select num_batch from unobserved nodes
    if len(selected_nodes) < num_batch:
        # get unobserved nodes that are not already in selected_nodes
        unobserved_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == -1 and node not in selected_nodes]
        # randomly select unobserved_nodes where available
        selected_nodes = np.concatenate((selected_nodes, rng.choice(unobserved_nodes, min(num_batch - len(selected_nodes), len(unobserved_nodes)), replace=False).astype(int)))

    return selected_nodes

def _preferential_reactive_select(G, observed_infection_states, last_observed_nodes, prefer_infected, rng, num_batch=1):
    """
    Similar to reactive_select, but preferentially selects infected nodes if prefer_infected is True, otherwise
    preferentially selects uninfected nodes.
    """
    # get infected (if prefer_infected is True, otherwise uninfected) nodes from last_observed_nodes
    relevant_last_observed_nodes = [node for node in last_observed_nodes if observed_infection_states[node] == (1 if prefer_infected else 0)]
    # get neighbors of relevant_last_observed_nodes
    relevant_last_observed_neighbors = list(set([neighbor for node in relevant_last_observed_nodes for neighbor in G.neighbors(node)]))
    # get unobserved neighbors of relevant_last_observed_nodes
    unobserved_relevant_last_observed_neighbors = [neighbor for neighbor in relevant_last_observed_neighbors if observed_infection_states[neighbor] == -1]
    # randomly select num_batch from unobserved_relevant_last_observed_neighbors
    selected_nodes = rng.choice(unobserved_relevant_last_observed_neighbors, min(num_batch, len(unobserved_relevant_last_observed_neighbors)), replace=False)
    # if not enough unobserved_relevant_last_observed_neighbors, randomly select num_batch from unobserved nodes
    if len(selected_nodes) < num_batch:
        # get unobserved nodes that are not already in selected_nodes
        unobserved_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == -1 and node not in selected_nodes]
        # randomly select num_batch - len(selected_nodes) from unobserved_nodes
        selected_nodes = np.concatenate((selected_nodes, rng.choice(unobserved_nodes, min(num_batch - len(selected_nodes), len(unobserved_nodes)), replace=False)))
        
    return selected_nodes

def reactive_infected_select(**kwargs):
    """
    Randomly selects num_batch unobserved nodes that are connected to newly observed nodes and preferentially selects infected nodes.
    """
    # extract required arguments
    required_keys = ['env_graph', 'surrogate', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)
    
    # get last_observed_nodes and last_observed_nodes from surrogate, num_batch, and env_graph
    last_observed_nodes = kwargs['surrogate'].get_last_observed_nodes()
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    num_batch = kwargs['num_batch']
    env_graph = kwargs['env_graph']
    rng = kwargs['rng']

    # preferentially select infected nodes
    return _preferential_reactive_select(env_graph, observed_infection_states, last_observed_nodes, True, rng, num_batch)

def reactive_infected_select_v2(**kwargs):
    """
    If there are observed infected nodes, randomly select num_batch unobserved nodes that are connected to them. Otherwise, randomly select num_batch unobserved nodes.
    """
    # extract required arguments
    required_keys = ['env_graph', 'surrogate', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)
    
    # get last_observed_nodes and last_observed_nodes from surrogate, num_batch, and env_graph
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    num_batch = kwargs['num_batch']
    env_graph = kwargs['env_graph']
    rng = kwargs['rng']

    # get observed infected nodes and all their neighbors
    observed_infected_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == 1]
    observed_infected_neighbors = list(set([neighbor for node in observed_infected_nodes for neighbor in env_graph.neighbors(node)]))
    # get unobserved nodes that are connected to observed_infected_nodes
    nb_unobserved_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == -1 and node in observed_infected_neighbors]
    # if neighbours of observed infected nodes are available, randomly select num_batch from them
    if len(nb_unobserved_nodes) > 0:
        selected_nodes = rng.choice(nb_unobserved_nodes, min(num_batch, len(nb_unobserved_nodes)), replace=False)
        # if fewer than num_batch nodes are available, randomly select the remaining nodes from unobserved_nodes
        available_unobserved_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == -1 and node not in selected_nodes]
        if len(selected_nodes) < num_batch:
            selected_nodes = np.concatenate((selected_nodes, rng.choice(available_unobserved_nodes, min(num_batch - len(selected_nodes), len(available_unobserved_nodes)), replace=False)))
    else:
        # randomly select num_batch unobserved nodes
        unobserved_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == -1]
        selected_nodes = rng.choice(unobserved_nodes, min(num_batch, len(unobserved_nodes)), replace=False)

    return selected_nodes

def reactive_uninfected_select(**kwargs):
    """
    Randomly selects num_batch unobserved nodes that are connected to newly observed nodes and preferentially selects infected nodes.
    """
    # extract required arguments
    required_keys = ['env_graph', 'surrogate', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)
    
    # get last_observed_nodes and last_observed_nodes from surrogate, num_batch, and env_graph
    last_observed_nodes = kwargs['surrogate'].get_last_observed_nodes()
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    num_batch = kwargs['num_batch']
    env_graph = kwargs['env_graph']
    rng = kwargs['rng']

    # preferentially select infected nodes
    return _preferential_reactive_select(env_graph, observed_infection_states, last_observed_nodes, False, rng, num_batch)

def _k_hop_neighbors_select(G, observed_infection_states, rng, k_hop=1, prefer_infected=True, num_batch=1):
    """
    Select num_batch nodes that are closest to the most number of infected/uninfected nodes within k-hop neighborhood.
    """
    # function to get all nodes within k_hop from a given node
    def get_k_hop_neighbors(node, k_hop):
        neighbors = set([node])
        for _ in range(k_hop):
            neighbors = set([neighbor for node in neighbors for neighbor in G.neighbors(node)]) | neighbors
        return neighbors

    # get unobserved nodes
    unobserved_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == -1]
    # get infected (if prefer_infected is True, otherwise uninfected) nodes
    nodes_of_interest = [node for node, infection_state in enumerate(observed_infection_states) if infection_state == (1 if prefer_infected else 0)]
    # for each unobserved node, get the number of nodes of interest within k-hop neighborhood
    unobserved_node_neighbors_of_interest = np.array([[node, len(get_k_hop_neighbors(node, k_hop) & set(nodes_of_interest))] for node in unobserved_nodes])
    # shuffle unobserved_node_neighbors_of_interest
    (rng if rng else np.random).shuffle(unobserved_node_neighbors_of_interest)
    # sort unobserved_node_neighbors_of_interest by the number of nodes of interest within k-hop neighborhood
    sorted_unobserved_node_neighbors_of_interest = unobserved_node_neighbors_of_interest[unobserved_node_neighbors_of_interest[:,1].argsort()[::-1]]
    # select num_batch nodes from sorted_unobserved_node_neighbors_of_interest
    selected_nodes = sorted_unobserved_node_neighbors_of_interest[:num_batch,0].astype(int)

    return selected_nodes

def k_hop_infected_neighbors_select(**kwargs):
    """
    Selects num_batch nodes that are closest to the most number of infected nodes within k-hop neighborhood.
    """
    # extract required arguments
    required_keys = ['env_graph', 'surrogate', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)
    
    # get last_observed_nodes and last_observed_nodes from surrogate, num_batch, and env_graph
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    num_batch = kwargs['num_batch']
    env_graph = kwargs['env_graph']
    k_hop = kwargs.get('k_hop', 2)
    rng = kwargs['rng']

    # select nodes that are closest to the most number of infected nodes within k-hop neighborhood
    return _k_hop_neighbors_select(env_graph, observed_infection_states, rng, k_hop=k_hop, prefer_infected=True, num_batch=num_batch)

def k_hop_uninfected_neighbors_select(**kwargs):
    """
    Selects num_batch nodes that are closest to the most number of uninfected nodes within k-hop neighborhood.
    """
    # extract required arguments
    required_keys = ['env_graph', 'surrogate', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)
    
    # get last_observed_nodes and last_observed_nodes from surrogate, num_batch, and env_graph
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    num_batch = kwargs['num_batch']
    env_graph = kwargs['env_graph']
    k_hop = kwargs.get('k_hop', 2)
    rng = kwargs['rng']

    # select nodes that are closest to the most number of uninfected nodes within k-hop neighborhood
    return _k_hop_neighbors_select(env_graph, observed_infection_states, rng, k_hop=k_hop, prefer_infected=False, num_batch=num_batch)