from .centrality_policies import *
from .heuristics import *
from .entropy_policies import *
from .local_entropy import local_entropy_select

class ActivePolicy:

    ############################ class params ############################

    # default available allocation-policies
    DEFAULT_ALLOCATION_POLICIES = {
        'random': random_select,
        # centrality-based heuristic methods
        'degree_central': degree_centrality_select,
        'degree_peripheral': degree_centrality_reverse_select,
        'betweenness_central': betweenness_centrality_select,
        'betweenness_peripheral': betweenness_centrality_reverse_select,
        'closeness_central': closeness_centrality_select,
        'closeness_peripheral': closeness_centrality_reverse_select,
        'eigenvector_central': eigenvector_centrality_select,
        'eigenvector_peripheral': eigenvector_centrality_reverse_select,
        'pagerank_central': pagerank_centrality_select,
        'pagerank_peripheral': pagerank_centrality_reverse_select,
        # heuristic methods
        'reactive': reactive_select,
        'reactive_infected': reactive_infected_select,
        'reactive_infected_v2': reactive_infected_select_v2,
        'reactive_uninfected': reactive_uninfected_select,
        'k2_hop_infected_neighbors': k_hop_infected_neighbors_select,
        'k2_hop_uninfected_neighbors': k_hop_uninfected_neighbors_select,
        # entropy-based methods
        'least_confidence': least_confidence_select,
        'node_entropy': node_entropy_select,
        'bald': bald_select,
        'bald_t2': bald_t2_select,
        # local-entropy
        'local_entropy': local_entropy_select
    }
    
    ############################ initialisations ############################

    def __init__(self, policy='random', num_batch=1):
        self.policy = policy
        self.num_batch = num_batch
        self.custom_policies = {}

        # check if policy is valid
        if self.policy not in self.DEFAULT_ALLOCATION_POLICIES:
            raise ValueError('policy must be one of {}'.format(self.DEFAULT_ALLOCATION_POLICIES.keys()))
        
        # check if num_batch is a positive integer
        if not (self.num_batch > 0 and isinstance(self.num_batch, int)):
            raise ValueError('num_batch must be a positive integer')
        
    ############################ utilities ############################

    def set_policy(self, policy):
        """Sets allocation-policy."""
        # check if policy is valid
        if policy not in self.DEFAULT_ALLOCATION_POLICIES and policy not in self.custom_policies:
            raise Exception('policy must be one of: %s' % ', '.join(list(self.DEFAULT_ALLOCATION_POLICIES.keys()) + list(self.custom_policies.keys())))
        # set policy
        self.policy = policy

    def deploy_policy(self, env_graph, surrogate, rng):
        """Deploy specified allocation-policy."""
        # construct kwargs, adding any additional arguments
        kwargs = {
            'env_graph': env_graph,
            'surrogate': surrogate,
            'rng': rng,
            'num_batch': self.num_batch
        }

        # get selected nodes
        if self.policy in self.custom_policies:
            selected_nodes = self.custom_policies[self.policy](**kwargs)
        else:
            selected_nodes = self.DEFAULT_ALLOCATION_POLICIES[self.policy](**kwargs)

        # return selected nodes
        return selected_nodes
    
    def add_custom_policy(self, policy_name, policy_func, set_policy=True):
        """Add custom allocation-policy."""
        # check if policy with the same name already exists
        if policy_name in self.DEFAULT_ALLOCATION_POLICIES or policy_name in self.custom_policies:
            raise ValueError('%s already exists as an available policy, please consider another name' % policy_name)
        # add custom policy
        self.custom_policies[policy_name] = policy_func
        # set policy to custom policy if set_policy is True
        if set_policy:
            self.set_policy(policy_name)
