from numpyro.diagnostics import hpdi
import jax.numpy as jnp
import numpy as np
import networkx as nx
import arviz as az
import json
import os

############################ graph-related helper functions ############################

def load_graph(filepath):
    """
    Reads graph from file.
    Allowed file formats: .adjlist, .edgelist, .graphml, .json
    """
    # check if file format is allowed
    allowed_file_formats = ['.adjlist', '.edgelist', '.graphml', '.json']
    # get file extension
    file_extension = os.path.splitext(filepath)[1]
    if file_extension not in allowed_file_formats:
        raise Exception('File format not allowed, please check')
    # read graph from file
    if file_extension == '.adjlist':
        G = nx.read_adjlist(filepath)
    elif file_extension == '.edgelist':
        G = nx.read_edgelist(filepath)
    elif file_extension == '.graphml':
        G = nx.read_graphml(filepath)
    elif file_extension == '.json':
        G = nx.node_link_graph(json.load(open(filepath)))

    return G

def save_graph(G, out_dir, filename, format='adjlist'):
    """
    Saves graph to file.
    Allowed file formats: .adjlist, .edgelist, .graphml, .json
    """
    # check if file format is allowed
    allowed_file_formats = ['adjlist', 'edgelist', 'graphml', 'json']
    format = format.lower()
    if format not in allowed_file_formats:
        raise Exception('File format not allowed, please check')
    # check that filename has the correct extension; if not, add it
    filename = filename if filename.endswith('.%s' % format) else ('%s.%s' % (filename, format))
    # check if out_dir exists
    os.makedirs(out_dir, exist_ok=True)
    # save graph to file
    if format == 'adjlist':
        nx.write_adjlist(G, os.path.join(out_dir, filename))
    elif format == 'edgelist':
        nx.write_edgelist(G, os.path.join(out_dir, filename))
    elif format == 'graphml':
        nx.write_graphml(G, os.path.join(out_dir, filename))
    elif format == 'json':
        json.dump(nx.node_link_data(G), open(os.path.join(out_dir, filename), 'w+'))

def generate_random_graph(node_num, edge_prob, seed=None):
    """Generates a random graph with node_num nodes and edge_prob edge probability."""
    # ensure that graph is connected
    while True:
        G = nx.gnp_random_graph(node_num, edge_prob, seed=seed)
        if nx.is_connected(G):
            break
    return G

def generate_random_graph_with_communities(community_sizes, intra_p, inter_p):
    """
    Generate a graph with a given number of communities, size of each community,
    probability of intra-community edges, and probability of inter-community edges.
    """
    # check if num_communities is valid
    if len(community_sizes) < 2:
        raise ValueError("There must be at least two communities")

    # set up empty graph
    G = nx.Graph()
    start = 0

    # add nodes and intra-community edges
    for size in community_sizes:
        community_nodes = range(start, start + size)
        G.add_nodes_from(community_nodes)
        G.add_edges_from(
            (u, v) for u in community_nodes for v in community_nodes if u != v and np.random.rand() < intra_p
        )
        start += size

    # add inter-community edges
    community_start = 0
    for i, size_i in enumerate(community_sizes):
        for j in range(i + 1, len(community_sizes)):
            community_nodes_i = range(community_start, community_start + size_i)
            community_start_j = sum(community_sizes[:j])
            community_nodes_j = range(community_start_j, community_start_j + community_sizes[j])
            G.add_edges_from(
                (u, v) for u in community_nodes_i for v in community_nodes_j if np.random.rand() < inter_p
            )
        community_start += size_i

    # ensure that the graph is connected
    components = list(nx.connected_components(G)) # find connected components
    while len(components) > 1:
        # pick one node from each of the first two components and connect them
        u = np.random.choice(list(components[0]))
        v = np.random.choice(list(components[1]))
        G.add_edge(u, v)
        components = list(nx.connected_components(G))

    return G

def get_adj_matrix(G):
    """Extracts adjacency matrix from a networkx graph object."""
    adj_matrix = nx.adjacency_matrix(G).todense()
    return adj_matrix

def get_degree_matrix(G):
    """Extracts degree matrix from a networkx graph object."""
    degree_matrix = nx.degree(G)
    return degree_matrix

def get_node_num(G):
    """Extracts number of nodes from a networkx graph object."""
    node_num = nx.number_of_nodes(G)
    return node_num

def get_edge_num(G):
    """Extracts number of edges from a networkx graph object."""
    edge_num = nx.number_of_edges(G)
    return edge_num

def check_infections_history(infections_history):
    """Checks if infections_history is valid."""
    # check if available times are consecutive
    times = sorted(infections_history.keys())
    if max(times) - min(times) + 1 != len(times):
        raise Exception('Available times are not consecutive')
    # check if infection_states are valid
    infection_state_dim = len(infections_history[times[0]])
    for time in times:
        infection_states = infections_history[time]
        if not all(infection_state in [0, 1] for infection_state in infection_states):
            raise Exception('Infection states must be either 0 or 1')
        if not all(len(infection_states) == infection_state_dim for infection_states in infections_history.values()):
            raise Exception('Infection states must have the same dimension across all times')

def load_infections_history(filepath, header=True):
    """Reads infections_history from file. Allowed file formats: .csv"""
    # check if file format is allowed
    allowed_file_formats = ['.csv']
    # get file extension
    file_extension = os.path.splitext(filepath)[1]
    if file_extension not in allowed_file_formats:
        raise Exception('File format not allowed, please check')
    # read infections_history from file
    with open(filepath, 'r') as infile:
        if header: # read and discard first line
            infile.readline()
        # iterate through lines
        line = infile.readline()
        infections_history = {}
        while line:
            # parse line
            time, *infection_states = line.strip().split(',')
            # add infection_states to infections_history
            infections_history[int(time)] = [int(infection_state) for infection_state in infection_states]
            # read next line
            line = infile.readline()

    # check if infections_history is valid
    check_infections_history(infections_history)

    return infections_history        

############################ MCMC-related helper functions ############################

def get_mcmc_p(mcmc, node=None, var=None, hpdi_ps=[0.9, 0.95]):
    """Extracts posterior samples of p from MCMC object."""
    samples = mcmc.get_samples()
    p_samples = samples['p']

    # get mean of each node
    if not var:
        return p_samples[:,node] if node and node < len(p_samples) else p_samples.transpose()
    elif var.lower() == 'mean':
        p_means = jnp.mean(p_samples, axis=0)
        return p_means[node] if node and node < len(p_means) else p_means
    elif var.lower() == 'hpdi':
        if node and node < len(p_samples):
            p_hpdis = { hpdi_p: hpdi(p_samples, hpdi_p)[node] for hpdi_p in hpdi_ps }
        else:
            p_hpdis = { hpdi_p: hpdi(p_samples, hpdi_p) for hpdi_p in hpdi_ps }
        return p_hpdis

def save_mcmc_to_netcdf(mcmc, out_dir, filename):
    """Saves MCMC object to netcdf file."""
    filepath = os.path.join(out_dir, filename)
    # check if file exists
    if os.path.isfile(filepath):
        raise Exception('File already exists, please check')
    # create directory if out_dir doesn't already exist
    os.makedirs(out_dir, exist_ok=True)
    # create inference_data
    inference_data = az.from_numpyro(mcmc)
    az.to_netcdf(inference_data, filepath)

def save_mcmc_posterior_mean(mcmc, out_dir, filename, with_hpdi=False, hpdi_ps=[0.9, 0.95]):
    """Saves posterior mean of p from MCMC object to file."""
    filepath = os.path.join(out_dir, filename)
    # check if file exists
    if os.path.isfile(filepath):
        raise Exception('File already exists, please check')
    # create directory if out_dir doesn't already exist
    os.makedirs(out_dir, exist_ok=True)
    # get posterior mean of p
    p_means = get_mcmc_p(mcmc, var='mean')
    # get hpdi of p if with_hpdi is True
    if with_hpdi:
        p_hpdis = get_mcmc_p(mcmc, var='hpdi', hpdi_ps=hpdi_ps)
    # save posterior mean of p to file
    with open(filepath, 'w+') as outfile:
        outfile.write('node,p_mean%s\n' % ((',' + ','.join(['p_hpdi%s_lw,p_hpdi%s_up' % (str(hpdi_p), str(hpdi_p)) for hpdi_p in hpdi_ps])) if with_hpdi else '')) # write header
        for node in range(len(p_means)):
            if with_hpdi:
                outfile.write('%d,%f,%s\n' % (node, p_means[node], ','.join(['%s,%s' % (str(p_hpdis[hpdi_p][0][node]), str(p_hpdis[hpdi_p][1][node])) for hpdi_p in hpdi_ps])))
            else:
                outfile.write('%d,%f\n' % (node, p_means[node]))
                              
############################ others ############################

def extract_kwargs(kwargs, required_keys):
    """Validates that all required keys are provided in kwargs and extract them."""
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f'{key} must be provided')
        
    return { key: kwargs[key] for key in required_keys }