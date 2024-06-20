from utils.helper_functions import get_mcmc_p, extract_kwargs
import numpy as np
import heapq

def least_confidence_select(**kwargs):
    """
    Select n unobserved nodes with most uncertain infection_estimates, i.e. estimate of latent probability p of infection states closest to 0.5.
    """
    # extract required arguments
    required_keys = ['surrogate', 'num_batch']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get observed_infection_states, infection_estimates from surrgate and num_batch
    observed_infection_states = kwargs['surrogate'].get_observed_infection_states()
    infection_estimates = kwargs['surrogate'].get_infection_estimates()
    num_batch = kwargs['num_batch']

    # extract posterior means
    p_means = get_mcmc_p(infection_estimates, var='mean')
    # calculate distance from 0.5 for each unobserved node
    unobserved_infection_estimate_distances = [(abs(p_means[node] - 0.5), node) for node, infection_state in enumerate(observed_infection_states) if infection_state == -1]
    # use nsmallest to get n nodes with smallest distance from 0.5
    selected_nodes = heapq.nsmallest(num_batch, unobserved_infection_estimate_distances)

    return [node for _, node in selected_nodes]

def node_entropy_select(**kwargs):
    """Select n unobserved nodes with highest node-specific entropy."""
    # extract required arguments
    required_keys = ['surrogate', 'num_batch']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get unobserved_nodes, infection_estimates from surrgate and num_batch
    unobserved_nodes = kwargs['surrogate'].get_unobserved_nodes()
    infection_estimates = kwargs['surrogate'].get_infection_estimates()
    num_batch = kwargs['num_batch']

    # extract posterior means
    p_means = get_mcmc_p(infection_estimates, var='mean')

    # compute entropy for each unobserved node
    p_y0 = np.array(p_means)[unobserved_nodes]
    p_y1 = 1 - p_y0
    nodes_entropy = -p_y0*np.log(p_y0) - p_y1*np.log(p_y1)

    # return first n nodes with highest entropy
    selected_nodes = np.array(unobserved_nodes)[np.array(np.argsort(nodes_entropy, kind='stable')[-num_batch:])]

    return list(selected_nodes)

def weighted_bald_select(**kwargs):
    """
    Select n unobserved nodes with maximum decrease in expected posterior entropy.
    t1_weight can be used to control the weight of the first term in the BALD score.
    """
    # extract required arguments
    required_keys = ['surrogate', 'num_batch']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get unobserved_nodes, infection_estimates from surrgate and num_batch
    unobserved_nodes = kwargs['surrogate'].get_unobserved_nodes()
    infection_estimates = kwargs['surrogate'].get_infection_estimates()
    num_batch = kwargs['num_batch']
    t1_weight = kwargs.get('t1_weight', 0.5)

    # extract posterior means
    p_means = get_mcmc_p(infection_estimates, var='mean')
    # extract posterior samples of p
    p_samples = get_mcmc_p(infection_estimates)

    # first term of BALD
    t1_p_y0 = np.array(p_means)[unobserved_nodes]
    t1_p_y1 = 1 - t1_p_y0
    t1 = -t1_p_y0*np.log(t1_p_y0) - t1_p_y1*np.log(t1_p_y1)

    # second term of BALD
    t2_p_y0 = p_samples[unobserved_nodes,:]
    t2_p_y1 = 1 - t2_p_y0
    t2 = -t2_p_y0*np.log(t2_p_y0) - t2_p_y1*np.log(t2_p_y1)

    # compute BALD score for each node
    weighted_bald_scores = t1_weight * t1 - (1-t1_weight) * np.nanmean(t2, axis=1)
    # return first n nodes with highest BALD scores
    selected_nodes = np.array(unobserved_nodes)[np.array(np.argsort(weighted_bald_scores, kind='stable')[-num_batch:])]

    return list(selected_nodes)

def bald_select(**kwargs):
    """Select n unobserved nodes with maximum decrease in expected posterior entropy (t1_weight=0.5)."""
    # standard BALD with t1_weight=0.5
    kwargs['t1_weight'] = 0.5

    return weighted_bald_select(**kwargs)

def bald_t2_select(**kwargs):
    """Select n unobserved nodes with minimum average posterior entropy (t1_weight=0)."""
    # only consider second term of BALD
    kwargs['t1_weight'] = 0

    return weighted_bald_select(**kwargs)