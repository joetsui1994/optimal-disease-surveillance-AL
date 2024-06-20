from binary_prevalence.surrogate.static_car import StaticCAR
import numpy as np
import copy
import uuid

AGENT_UUID_LENGTH = 16

class BaseAgent:

    ############################ initialisations ############################

    def __init__(self, env_graph, passive_allocation_policy, allocation_policy, surrogate=None, allocation_start_time=0, query_rate=1):
        self._id = uuid.uuid4()
        self._rng = np.random.default_rng(self._id.int % (2**32))
        self.env_graph = env_graph
        self.surrogate = surrogate
        self.passive_allocation_policy = passive_allocation_policy
        self.allocation_policy = allocation_policy
        self.allocation_start_time = allocation_start_time # time to perform first (passive) allocation, in integer units of time in env
        self.query_rate = query_rate # in units of time in env; e.g., if query_rate = 0.5, then agent queries env every 2 time-steps (env)
        self._is_run = False

        # check that query_rate is positive
        if not self.query_rate > 0:
            raise ValueError('query_rate must be positive')

    @property
    def id(self):
        """Gets agent id."""
        return str(self._id)[:AGENT_UUID_LENGTH]
    
    def reset_id(self):
        """Resets agent id and rng."""
        self._id = uuid.uuid4()
        self._rng = np.random.default_rng(self._id.int % (2**32))

    @property
    def is_run(self):
        """Gets whether agent has been ran."""
        return self._is_run
    
    def set_run(self):
        """Sets whether agent has been ran."""
        self._is_run = True
    
    def reset_run(self):
        """Resets whether agent has been ran."""
        self._is_run = False
        self.surrogate.reset(keep_env=True, keep_configs=True)
        # reset rng
        self._rng = np.random.default_rng(self._id.int % (2**32))

    ############################ main ############################

    def deploy_passive_allocation_policy(self):
        """Deploy passive allocation policy."""
        selected_nodes = self.passive_allocation_policy.deploy_policy(self.env_graph, self._rng)
        return selected_nodes
    
    def deploy_allocation_policy(self):
        """Deploy allocation policy."""
        selected_nodes = self.allocation_policy.deploy_policy(self.env_graph, self.surrogate, self._rng)
        return selected_nodes
    
    def update_surrogate(self, time, new_observed_infection_states):
        """Update surrogate with new observed infection states from the environment."""
        self.surrogate.update_observed_infection_states(time, new_observed_infection_states)
        # update surrogate if surrogate model is not up to date
        if not self.surrogate.is_model_updated():
            self.surrogate.update_model(time)

    ############################ utilities ############################
    
    def add_surrogate(self, model='car'):
        """
        Add surrogate model to agent. This is recommended as it ensures that the surrogate
        is set up with the same env_graph as the agent.
        """
        if model != 'car':
            raise ValueError('Only CAR model is currently supported')
        self.surrogate = StaticCAR(self.env_graph, rng_seed=self._id.int % (2**32))

    def spawn_empty_copies(self, num_copies):
        """Spawns num_copies empty copies of itself."""
        agent_copies = [copy.deepcopy(self) for _ in range(num_copies)]
        # reset surrogate of each agent copy
        for agent_copy in agent_copies:
            agent_copy.reset_id()
            agent_copy.reset_run()

        return agent_copies
    
