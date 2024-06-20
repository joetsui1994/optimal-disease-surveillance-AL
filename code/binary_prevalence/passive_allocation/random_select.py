from utils.helper_functions import extract_kwargs

def random_select(**kwargs):
    """Randomly selects num_batch nodes."""
    # extract required arguments
    required_keys = ['env_graph', 'num_batch', 'rng']
    kwargs = extract_kwargs(kwargs, required_keys)

    # get nodes from env and num_batch
    nodes = list(kwargs['env_graph'].nodes)
    num_batch = kwargs['num_batch']
    rng = kwargs['rng']

    # randomly select n from nodes
    selected_nodes = rng.choice(nodes, min(num_batch, len(nodes)), replace=False)

    return selected_nodes