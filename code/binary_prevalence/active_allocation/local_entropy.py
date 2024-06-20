from utils.helper_functions import get_mcmc_p, extract_kwargs
import networkx as nx
import numpy as np

def local_entropy_select(**kwargs):
    """
    Select n unobserved nodes with maximum expected impact, as measured by the distance-weighted average entropy of all nodes in the graph.
    The distance weight is computed as the inverse of the geodesic distance from the candidate node to each observed node, raised to the power of the gamma parameter.
    """
    # extract required arguments
    required_keys = ['surrogate', 'num_batch', 'env_graph']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get env_graph from enviornment, unobserved_nodes from surrgate and num_batch
    unobserved_nodes = kwargs['surrogate'].get_unobserved_nodes()
    env_graph = kwargs['env_graph']
    num_batch = kwargs['num_batch']

    # extract posterior means
    infection_estimates = kwargs['surrogate'].get_infection_estimates()
    p_means = get_mcmc_p(infection_estimates, var='mean')
    entropies = -p_means * np.log(p_means) - (1 - p_means) * np.log(1 - p_means)

    # calculate geodesic distances between all nodes
    geodesic_distances = dict(nx.all_pairs_shortest_path_length(env_graph))

    # calculate distance weights
    gamma = 1
    distance_weights = { node: 1 / np.array([distances[other] if node != other else 1 for other in range(len(geodesic_distances))]) ** gamma
                        for node, distances in geodesic_distances.items() }

    # consider each unobserved node as a candidate and score it according to the sum of distance-weighted distance from 0.5 (summed over all other unobserved nodes)
    candidate_scores = {}
    for node in unobserved_nodes:
        candidate_score = np.sum(entropies * distance_weights[node]) - entropies[node]
        normalisation = np.sum(distance_weights[node]) - distance_weights[node][node]
        candidate_scores[node] = candidate_score / normalisation

    # select num_batch_candidates with the highest scores
    selected_nodes = [node for node, score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)][:num_batch]

    return selected_nodes