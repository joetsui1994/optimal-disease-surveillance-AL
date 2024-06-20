from .random_select import random_select
from .centrality_policies import *

class PassivePolicy:

    ############################ class params ############################

    # default available passive-allocation-policies
    DEFAULT_PASSIVE_POLICIES = {
        'random': random_select,
        # centrality-based methods
        'degree_central': degree_centrality_select,
        'degree_peripheral': degree_centrality_reverse_select,
        'betweenness_central': betweenness_centrality_select,
        'betweenness_peripheral': betweenness_centrality_reverse_select,
        'closeness_central': closeness_centrality_select,
        'closeness_peripheral': closeness_centrality_reverse_select,
        'eigenvector_central': eigenvector_centrality_select,
        'eigenvector_peripheral': eigenvector_centrality_reverse_select,
        'pagerank_central': pagerank_centrality_select,
        'pagerank_peripheral': pagerank_centrality_reverse_select
    }
    
    ############################ initialisations ############################

    def __init__(self, policy='random', num_batch=1):
        self.policy = policy
        self.num_batch = num_batch
        self.custom_policies = {}

        # check if policy is valid
        if self.policy not in self.DEFAULT_PASSIVE_POLICIES:
            raise ValueError('policy must be one of {}'.format(self.DEFAULT_PASSIVE_POLICIES.keys()))
        
        # check if num_batch is a positive integer
        if not (self.num_batch > 0 and isinstance(self.num_batch, int)):
            raise ValueError('num_batch must be a positive integer')
        
    ############################ utilities ############################

    def set_policy(self, policy):
        """Sets passive-allocation-policy."""
        # check if policy is valid
        if policy not in self.DEFAULT_PASSIVE_POLICIES and policy not in self.custom_policies:
            raise Exception('policy must be one of: %s' % ', '.join(list(self.DEFAULT_PASSIVE_POLICIES.keys()) + list(self.custom_policies.keys())))
        # set policy
        self.policy = policy

    def deploy_policy(self, env_graph, rng):
        """Deploy specified allocation-policy."""
        # construct kwargs, adding any additional arguments
        kwargs = {
            'env_graph': env_graph,
            'rng': rng,
            'num_batch': self.num_batch
        }

        # get selected nodes
        if self.policy in self.custom_policies:
            selected_nodes = self.custom_policies[self.policy](**kwargs)
        else:
            selected_nodes = self.DEFAULT_PASSIVE_POLICIES[self.policy](**kwargs)

        return selected_nodes
    
    def add_custom_policy(self, policy_name, policy_func, set_policy=True):
        """Add custom allocation-policy."""
        # check that policy_name is a string
        if not isinstance(policy_name, str):
            raise ValueError('policy_name must be a string')
        # check that policy_func is a function
        if not callable(policy_func):
            raise ValueError('policy_func must be a function')
        # check if policy with the same name already exists
        if policy_name in self.DEFAULT_PASSIVE_POLICIES or policy_name in self.custom_policies:
            raise ValueError('%s already exists as an available policy, please consider another name' % policy_name)
        # add custom policy
        self.custom_policies[policy_name] = policy_func
        # set policy to custom policy if set_policy is True
        if set_policy:
            self.set_policy(policy_name)
