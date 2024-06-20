from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from utils.helper_functions import get_mcmc_p, extract_kwargs
import numpy as np

def _auc_compute(**kwargs):
    """Calculate AUC-ROC."""
    # extract required arguments
    required_keys = ['time', 'infection_states_history', 'observed_infection_states_history', 'infection_estimates_history', 'pr']
    kwargs = extract_kwargs(kwargs, required_keys)

    # check if time is specified
    time = kwargs['time']
    times_to_evaluate = []
    if time is not None:
        if time not in kwargs['infection_estimates_history']:
            raise Exception('time must be one of: %s' % ', '.join(kwargs['infection_estimates_history'].keys()))
        times_to_evaluate.append(time)
    else:
        times_to_evaluate = list(kwargs['infection_estimates_history'].keys())

    # iterate over time-steps
    auc_history = {}
    for time_to_evaluate in times_to_evaluate:
        # get infection_estimates
        infection_estimates = kwargs['infection_estimates_history'][time_to_evaluate]
        # get only unobserved nodes
        unobserved_nodes_indices = np.where(np.array(kwargs['observed_infection_states_history'][time_to_evaluate]) == -1)[0]
        # take infection_states of only these unobserved nodes
        unobserved_infection_states = kwargs['infection_states_history'][time_to_evaluate][unobserved_nodes_indices]
        # take infection_estimates of only these unobserved nodes
        p_means = get_mcmc_p(infection_estimates, var='mean')
        unobserved_p_means = p_means[unobserved_nodes_indices]

        # check that there are at least two unique infection states
        if len(set(unobserved_infection_states)) < 2:
            auc_history[time_to_evaluate] = None
            continue

        # evaluate auc-score
        if kwargs['pr']:
            # calculate precision and recall for different threshold values
            precision, recall, _ = precision_recall_curve(unobserved_infection_states, unobserved_p_means)
            # calculate the AUC for the precision-recall curve
            auc_score = auc(recall, precision)
        else:
            auc_score = roc_auc_score(unobserved_infection_states, unobserved_p_means)

        # store
        auc_history[time_to_evaluate] = auc_score

    return auc_history

def calculate_auc(**kwargs):
    """Calculate AUC-ROC."""
    pr_kwargs = { **kwargs, 'pr': False }
    return _auc_compute(**pr_kwargs)

def calculate_pr_auc(**kwargs):
    """Calculate AUC for precision-recall curve."""
    pr_kwargs = { **kwargs, 'pr': True }
    return _auc_compute(**pr_kwargs)