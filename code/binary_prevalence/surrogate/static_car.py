import os
from utils.helper_functions import save_mcmc_to_netcdf, save_mcmc_posterior_mean
import numpyro
import jax.numpy as jnp
from jax.random import PRNGKey
import warnings
import networkx as nx
import numpy as np
import yaml
import sys
import io
from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    # Backup the original stderr
    original_stderr = sys.stderr
    try:
        # redirect stderr to nowhere
        sys.stderr = io.StringIO()
        yield
    finally:
        # restore the original stderr
        sys.stderr = original_stderr

class StaticCAR:

    ############################ class params ############################
   
    # default MCMC parameters
    DEFAULT_ALPHA_PARAMS = { 'fixed': -1, 'low': 0., 'high': 1. }
    DEFAULT_TAU_PARAMS = { 'prior': 'lognormal', 'mu': 0, 'sigma': 0.1, 'alpha': 3, 'beta': 2 }
    DEFAULT_MCMC_PARAMS = { 'num_warmup': 1000, 'num_samples': 2000, 'num_chains': 2, 'parallel_chains': False, 'num_devices': 2 }

    ############################ initialisations ############################

    def __init__(self, env_graph, rng_seed=None):
        self.rng = np.random.default_rng(rng_seed)
        self.env_graph = env_graph
        self.observed_infection_states_history = {} # observed infection states history
        self.infection_estimates_history = {} # posterior samples of latent probability p of infection states from mcmc
        self.configs = { 'alpha_params': StaticCAR.DEFAULT_ALPHA_PARAMS,
                        'tau_params': StaticCAR.DEFAULT_TAU_PARAMS,
                        'mcmc_params': StaticCAR.DEFAULT_MCMC_PARAMS }

    def reset_rng(self, rng_seed=None):
        """Resets random number generator."""
        self.rng = np.random.default_rng(rng_seed)

    def reset(self, rng_seed=None, keep_env=True, keep_configs=False):
        """Resets surrogate."""
        self.reset_rng(rng_seed=rng_seed)
        self.observed_infection_states_history = {}
        self.infection_estimates_history = {}
        if not keep_env:
            self.env_graph = None
        if not keep_configs:
            self.configs = None

    def parse_configs_yaml(self, yaml_file):
        """Parses surrogate configurations from YAML file."""
        # check if filepath exists
        if not os.path.exists(yaml_file):
            raise Exception('Specified file does not exist')
        # parse surrogate configurations from YAML file
        with open(yaml_file, 'r') as infile:
            configs = yaml.load(infile, Loader=yaml.FullLoader)
        # set surrogate configurations
        self.configs = { **self.configs, **configs }

    ############################ update model ############################

    @staticmethod
    def _CAR_model(args, observed_nodes, observed_infection_states=None):
        # expit function
        def expit(x):
            return 1/(1 + jnp.exp(-x))
        
        # extract required data and params
        adj_matrix = args['adj_matrix']
        node_num = args['node_num']
        alpha_params = args['alpha_params']
        tau_params = args['tau_params']

        # define priors
        b0 = numpyro.sample('b0', numpyro.distributions.Normal(0, 1).expand([node_num]))
        tau = numpyro.sample('tau', numpyro.distributions.LogNormal(tau_params['mu'], tau_params['sigma'])
                             if tau_params['prior'].lower() == 'lognormal' else numpyro.distributions.Gamma(tau_params['alpha'], tau_params['beta']))
        alpha = alpha_params['fixed'] if alpha_params['fixed'] > 0 else numpyro.sample('alpha', numpyro.distributions.Uniform(low=alpha_params['low'], high=alpha_params['high']))
        # define CAR prior
        car = numpyro.sample('car', numpyro.distributions.CAR(loc=jnp.zeros(node_num), correlation=alpha, adj_matrix=adj_matrix,
                                                              conditional_precision=tau, is_sparse=False))
        # define linear predictor
        lin_pred = b0 + car
        p = numpyro.deterministic('p', expit(lin_pred))

        # likelihood - only use countries with observations
        if observed_infection_states is not None:
            numpyro.sample("obs", numpyro.distributions.Bernoulli(p[observed_nodes]), obs=observed_infection_states[observed_nodes])

    def update_model(self, time):
        """Rebuilds CAR model and updates infection estimates history."""
        # extract MCMC parameters from kwargs, with values from DEFAULT_MCMC_PARAMS as default
        if self.configs is not None and 'mcmc_params' in self.configs:
            mcmc_params = { param: self.configs['mcmc_params'].get(param, value) for param, value in StaticCAR.DEFAULT_MCMC_PARAMS.items() }
        else:
            mcmc_params = StaticCAR.DEFAULT_MCMC_PARAMS
        # extract graph data from self.env_graph
        graph_data = {
            'adj_matrix': nx.adjacency_matrix(self.env_graph, nodelist=sorted(self.env_graph.nodes)).toarray(),
            'node_num': len(self.env_graph.nodes),
            'alpha_params': self.configs['alpha_params'],
            'tau_params': self.configs['tau_params']
        }
        # get most recent observed_infection_states
        observed_infection_states = self.get_observed_infection_states(time=time)
        # extract observed nodes
        observed_nodes = [node for node, infection_state in enumerate(observed_infection_states) if infection_state != -1]

        # inference
        with suppress_stderr():
            warnings.filterwarnings('ignore', category=DeprecationWarning, module="numpyro.distributions.util")
            numpyro.set_host_device_count(mcmc_params['num_devices'])
            rng_key = PRNGKey(self.rng.integers(0, 2**32 - 1))
            kernel = numpyro.infer.NUTS(StaticCAR._CAR_model)
            mcmc = numpyro.infer.MCMC(kernel, num_warmup=mcmc_params['num_warmup'], num_samples=mcmc_params['num_samples'], num_chains=mcmc_params['num_chains'],
                                        chain_method='parallel' if mcmc_params['parallel_chains'] else 'sequential', progress_bar=True)
            mcmc.run(rng_key, graph_data, observed_nodes=jnp.array(observed_nodes), observed_infection_states=jnp.array(observed_infection_states))
            warnings.filterwarnings('default')

        # save infection estimates
        self._update_infection_estimates(time, mcmc)

    def _update_infection_estimates(self, time, mcmc):
        """Updates infection estimates history."""
        # check if time is valid
        if time < self.get_time_of_last_update():
            raise Exception('time must be greater than or equal to time of last update')
        # update infection_estimates_history
        self.infection_estimates_history[time] = mcmc

    def update_observed_infection_states(self, time, new_observed_infection_states):
        """Updates observed infection states history."""
        # check if time is valid
        if time < self.get_time_of_last_observation():
            raise Exception('time must be greater than or equal to time of last observation')
        # update observed_infection_states_history
        self.observed_infection_states_history[time] = new_observed_infection_states

    ############################ I/O ############################

    def save_infection_estimates(self, out_dir, filename=None, time=None, point_estimate_only=False, with_hpdi=False, hpdi_ps=[0.9, 0.95]):
        """Saves infection estimates to netcdf file."""
        # set time to maximum recorded time if not specified
        time = self.get_time_of_last_update() if time is None else time
        # check if out_dir exists, otherwise create it
        os.makedirs(out_dir, exist_ok=True)
        # check if time is valid
        if not (time in self.infection_estimates_history):
            raise Exception("Specified time cannot be found in surrogate's infection_estimates_history")
        # get infection estimates at specified time
        infection_estimates = self.get_infection_estimates(time=time)
        # default filename
        filename = ('infection_point_estimates_t{}.csv'.format(time) if point_estimate_only else 'infection_estimates_t{}.nc'.format(time)) if filename is None else filename
        # save infection estimates or point estimates
        if point_estimate_only:
            save_mcmc_posterior_mean(infection_estimates, out_dir, filename, with_hpdi=with_hpdi, hpdi_ps=hpdi_ps)
        else:
            save_mcmc_to_netcdf(infection_estimates, out_dir, filename)

    def save_observed_infection_states(self, out_dir, filename=None, time=None):
        """Saves observed infection states to csv file."""
        # set time to maximum recorded time if not specified
        time = self.get_time_of_last_observation() if time is None else time
        # check if out_dir exists, otherwise create it
        os.makedirs(out_dir, exist_ok=True)
        # check if time is valid
        if not (time in self.observed_infection_states_history):
            raise Exception("Specified time cannot be found in surrogate's observed_infection_states_history")
        # default filename
        filename = 'observed_infection_states_t{}.csv'.format(time) if filename is None else filename
        # save observed infection states to csv file
        with open(os.path.join(out_dir, filename), 'w+') as outfile:
            outfile.write('time,%s\n' % ','.join([str(node) for node in range(len(self.env_graph.nodes))]))
            # get sorted times (before time)
            sorted_times = sorted([t for t in self.observed_infection_states_history.keys() if t <= time])
            # write infections_history from early to late
            for time in sorted_times:
                outfile.write('%d,%s\n' % (time, ','.join([str(infection_states) for infection_states in self.observed_infection_states_history[time]])))

    ############################ utilities ############################

    def purge_infection_estimates(self, time=None):
        """
        Purges infection estimates history up to specified time (keeping estimates at specified time).\
        If time is not specified, purge all infection estimates except for most recent estimates.
        """
        # set time to maximum recorded time if not specified
        time = self.get_time_of_last_update() if time is None else time
        # check if time is valid
        if time > self.get_time_of_last_update():
            raise Exception('time must be less than or equal to time of last update')
        # purge infection_estimates_history
        self.infection_estimates_history = { t: (None if t < time else infection_estimates) for t, infection_estimates in self.infection_estimates_history.items() }

    def get_time_of_last_update(self):
        """Returns time of last model update."""
        infection_estimates_last_update = max(self.infection_estimates_history.keys()) if self.infection_estimates_history else -1
        return infection_estimates_last_update
    
    def get_time_of_last_observation(self):
        """Returns time of last observation."""
        return max(self.observed_infection_states_history.keys()) if self.observed_infection_states_history else -1

    def is_model_updated(self):
        """Checks if model is updated by comparing time of last update and time of last observation."""
        return (self.get_time_of_last_update() == self.get_time_of_last_observation()) and (self.get_time_of_last_update() != -1)

    def get_observed_infection_states(self, time=None):
        """Returns observed infection states. If time is not specified, return most recent observed infection states."""
        # set time to maximum recorded time if not specified
        time = self.get_time_of_last_observation() if time is None else time
        # before first observation
        if time == -1:
            return None
        # check if time is valid
        if time > self.get_time_of_last_observation():
            raise Exception('Requested time is greater than time of last observation')
        # check if time exists in observed_infection_states_history
        if time not in self.observed_infection_states_history:
            raise Exception('observed_infection_states does not exist for time {}'.format(time))
        # return observed_infection_states
        return self.observed_infection_states_history[time]
    
    def get_infection_estimates(self, time=None):
        """Returns infection estimates. If time is not specified, return most recent infection estimates."""
        # set time to maximum recorded time if not specified
        time = self.get_time_of_last_update() if time is None else time
        # before first model update
        if time == -1:
            return None
        # check if time is valid
        if time > self.get_time_of_last_update():
            raise Exception('Requested time is greater than time of last update')
        # check if time exists in infection_estimates_history
        if time not in self.infection_estimates_history:
            raise Exception('infection_estimates does not exist for time {}'.format(time))
        # return infection_estimates
        return self.infection_estimates_history[time]

    def get_last_observed_nodes(self, time=None):
        """Returns last observed nodes. If time is not specified, return most recent last observed nodes."""
        # set time to maximum recorded time if not specified
        time = self.get_time_of_last_observation() if time is None else time
        # before first observation
        if time == -1:
            return None
        # check if time is valid
        if time > self.get_time_of_last_observation():
            raise Exception('Requested time is greater than time of last observation')
        # check if time exists in observed_infection_states_history
        if time not in self.observed_infection_states_history:
            raise Exception('observed_infection_states does not exist for time {}'.format(time))
        # if time is 0, return all observed nodes
        if time == 0:
            last_observed_nodes = [node for node, infection_state in enumerate(self.observed_infection_states_history[0]) if infection_state != -1]
        else:
            # get sorted times
            sorted_times = sorted(list(self.observed_infection_states_history.keys()))
            # get observed_infection_states at specified time
            curr_observed_infection_states = self.get_observed_infection_states(time=time)
            # get observed_infection_states at previous time
            prev_time = sorted_times[sorted_times.index(time) - 1]
            prev_observed_infection_states = self.get_observed_infection_states(time=prev_time)
            # combine
            compare_observed_infection_states = zip(prev_observed_infection_states, curr_observed_infection_states)
            
            # compare prev_observed_infection_states and curr_observed_infection_states to get most recent observations
            last_observed_nodes = [node for node, (prev_infection_state, curr_infection_state) in enumerate(compare_observed_infection_states)
                                   if (prev_infection_state == -1 and curr_infection_state != -1)]
            
        return last_observed_nodes
    
    def get_unobserved_nodes(self, time=None):
        """Returns unobserved nodes. If time is not specified, return most recent unobserved nodes."""
        # set time to maximum recorded time if not specified
        time = self.get_time_of_last_observation() if time is None else time
        # before first model update
        if time == -1:
            return None
        # check if time is valid
        if time > self.get_time_of_last_observation():
            raise Exception('Requested time is greater than time of last observation')
        # get observed_infected_states at time
        observed_infected_states = self.get_observed_infection_states(time=time)
        unobserved_nodes = [node for node, infection_state in enumerate(observed_infected_states) if infection_state == -1]

        return unobserved_nodes