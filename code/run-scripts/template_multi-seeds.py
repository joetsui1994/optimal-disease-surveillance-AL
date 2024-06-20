#!/usr/bin/env python
# coding: utf-8

# In[1]:


# project_dir
project_dir = <absolute_path_to_project_dir_on_server>
prod = # Set to True if production, False if testing


# In[2]:


# load in modules
import os
import sys
import numpy as np
import networkx as nx
if prod:
    sys.path.append(project_dir) # add the path to modules
else:
    sys.path.append(<absolute_path_to_project_dir_on_local_machine>) # add the path to modules
from binary_prevalence.active_allocation import ActivePolicy
from binary_prevalence.passive_allocation import PassivePolicy
from binary_prevalence.agent import BaseAgent
from binary_prevalence.evaluation import BaseEvaluator
from binary_prevalence.environment import BinaryPrevalenceEnvironment
from tao_manager import TaoManager


# #### Initialise environment from exported graph and simulated static prevalence

# In[3]:


# read in all simulated outbreak data
num_random_initial_nodes = 25 # number of random initial nodes to run
outbreak_start_index = 0 # inclusive
outbreak_end_index = 10 # exclusive

simulated_outbreaks = {}
endp_dir = <relative_path_to_simulated_outbreaks_dir> # relative path to directory containing all simulated outbreaks within project_dir
# if production
if prod:
    endp_dir = os.path.join(project_dir, <relative_path_to_input_dir>, endp_dir) # relative path to parent directory of endp_dir within project_dir

# iterate over all directories in endp_dir
# seeds2run = [] # list of seeds (corresponding to different outbreaks) to run
# for index, entry in enumerate(seeds2run):
for index, entry in enumerate(os.listdir(endp_dir)):
    if index < outbreak_start_index or index >= outbreak_end_index:
        continue
    entry_path = os.path.join(endp_dir, entry)
    if os.path.isdir(entry_path):
        # get seed from directory name
        seed = int(entry.split('seed')[1])
        simulated_outbreaks[seed] = { 'index': index }
        for file in os.listdir(entry_path):
            if file.startswith('infection_states'):
                with open(os.path.join(entry_path, file), 'r') as f:
                    infection_states = [int(x) for x in f.read().split(',')]
                    simulated_outbreaks[seed]['infection_states'] = infection_states
            elif file.startswith('initial_nodelist'):
                with open(os.path.join(entry_path, file), 'r') as f:
                    initial_nodelist = [int(x) for x in f.read().split(',')]
                    simulated_outbreaks[seed]['initial_nodelist'] = initial_nodelist[:num_random_initial_nodes]


# In[4]:


# print all simulated outbreaks (seeds), these are the outbreaks we will be running in this run-script
print(simulated_outbreaks.keys())


# In[5]:


# read in graph as adjacency list
# if production
if prod:
    graph_dir = os.path.join(project_dir, <relative_path_to_graph_dir>) # relative path to directory containing graph within project_dir
else:
    graph_dir = <absolute_path_to_graph_dir_on_local_machine> # absolute path to directory containing graph on local machine
G = nx.read_adjlist(os.path.join(graph_dir, 'target_graph.relabelled.adjlist'))
# relabel nodes as integers
mapping = { node: int(node) for node in G.nodes }
G = nx.relabel_nodes(G, mapping)

for seed, outbreak_data in simulated_outbreaks.items():

    # initialise (binary) prevalence environment from final state
    env = BinaryPrevalenceEnvironment(graph=G)
    env.initialise_static_infection_states(outbreak_data['infection_states'], len(outbreak_data['infection_states']))

    ############################################################################
    # create passive-policy instances
    ############################################################################

    # initialise passive-policy instance (placeholder)
    passive_policy = PassivePolicy()
    passive_policy.num_batch = 1 # not necessary, only for bookkeeping ### CHANGE ###

    ############################################################################
    # create active-policy instances
    ############################################################################

    # initialise active-policy (random)
    active_random = ActivePolicy('random')
    active_random.num_batch = 1

    # initialise active-policy (reactive-infected)
    active_reactive_infected = ActivePolicy('reactive_infected_v2')
    active_reactive_infected.num_batch = 1

    # initialise active-policy (node_entropy)
    active_node_entropy = ActivePolicy('node_entropy')
    active_node_entropy.num_batch = 1

    # initialise active-policy (BALD)
    active_bald = ActivePolicy('bald')
    active_bald.num_batch = 1

    # initialise active-policy (pagerank_central)
    active_pagerank = ActivePolicy('pagerank_central')
    active_pagerank.num_batch = 1

    # initialise acvtive-policy (degree_central)
    active_degree_central = ActivePolicy('degree_central')
    active_degree_central.num_batch = 1

    # initialise acvtive-policy (local_entropy)
    active_local_entropy = ActivePolicy('local_entropy')
    active_local_entropy.num_batch = 1

    ############################################################################
    # create model agents
    ############################################################################

    # passive: random; active: random
    random_random_model_agent = BaseAgent(env.graph, passive_policy, active_random)
    random_random_model_agent.add_surrogate(model='car')

    # passive: random; active: reactive-infected
    random_reactive_infected_model_agent = BaseAgent(env.graph, passive_policy, active_reactive_infected)
    random_reactive_infected_model_agent.add_surrogate(model='car')

    # passive: random; active: node_entropy
    random_node_entropy_model_agent = BaseAgent(env.graph, passive_policy, active_node_entropy)
    random_node_entropy_model_agent.add_surrogate(model='car')

    # passive: random; active: BALD
    random_bald_model_agent = BaseAgent(env.graph, passive_policy, active_bald)
    random_bald_model_agent.add_surrogate(model='car')

    # passive: random; active: pagerank_central
    random_pagerank_model_agent = BaseAgent(env.graph, passive_policy, active_pagerank)
    random_pagerank_model_agent.add_surrogate(model='car')

    # passive: random; active: degree_central
    random_degree_central_model_agent = BaseAgent(env.graph, passive_policy, active_degree_central)
    random_degree_central_model_agent.add_surrogate(model='car')

    # passive: random; active: local_entropy
    random_local_entropy_model_agent = BaseAgent(env.graph, passive_policy, active_local_entropy)
    random_local_entropy_model_agent.add_surrogate(model='car')

    ############################################################################
    # set up surrogate config
    ############################################################################
    surrogate_configs = {
        'alpha_params': { 'fixed': 0.95, 'low': 0., 'high': 1. },
        'tau_params': { 'prior': 'lognormal', 'mu': 0, 'sigma': 0.1, 'alpha': 3, 'beta': 2 },
        'mcmc_params': { 'num_warmup': 200, 'num_samples': 800, 'num_chains': 3, 'parallel_chains': True, 'num_devices': 3 }
        }
    random_random_model_agent.surrogate.configs = surrogate_configs
    random_reactive_infected_model_agent.surrogate.configs = surrogate_configs
    random_node_entropy_model_agent.surrogate.configs = surrogate_configs
    random_bald_model_agent.surrogate.configs = surrogate_configs
    random_pagerank_model_agent.surrogate.configs = surrogate_configs
    random_degree_central_model_agent.surrogate.configs = surrogate_configs
    random_local_entropy_model_agent.surrogate.configs = surrogate_configs

    ############################################################################
    # spawn agent copies, each with a different initial node
    ############################################################################

    # spawn agent copies
    random_random_agent_copies = random_random_model_agent.spawn_empty_copies(len(outbreak_data['initial_nodelist']))
    random_reactive_infected_agent_copies = random_reactive_infected_model_agent.spawn_empty_copies(len(outbreak_data['initial_nodelist']))
    random_node_entropy_agent_copies = random_node_entropy_model_agent.spawn_empty_copies(len(outbreak_data['initial_nodelist']))
    random_bald_agent_copies = random_bald_model_agent.spawn_empty_copies(len(outbreak_data['initial_nodelist']))
    random_pagerank_agent_copies = random_pagerank_model_agent.spawn_empty_copies(len(outbreak_data['initial_nodelist']))
    random_degree_central_agent_copies = random_degree_central_model_agent.spawn_empty_copies(len(outbreak_data['initial_nodelist']))
    random_local_entropy_agent_copies = random_local_entropy_model_agent.spawn_empty_copies(len(outbreak_data['initial_nodelist']))

    # assign passive-policy to agent copies, each with a different initial node
    for index, initial_node in enumerate(outbreak_data['initial_nodelist']):

        # define passive-policy function
        def passive_fix_initial(initial=initial_node, **kwargs):
            return np.array([initial])
        
        # create temporary passive policy
        tmp_passive_policy = PassivePolicy()
        tmp_passive_policy.add_custom_policy('initial(%d)' % initial_node, passive_fix_initial)

        # assign to agent
        random_random_agent_copies[index].passive_allocation_policy = tmp_passive_policy
        random_reactive_infected_agent_copies[index].passive_allocation_policy = tmp_passive_policy
        random_node_entropy_agent_copies[index].passive_allocation_policy = tmp_passive_policy
        random_bald_agent_copies[index].passive_allocation_policy = tmp_passive_policy
        random_pagerank_agent_copies[index].passive_allocation_policy = tmp_passive_policy
        random_degree_central_agent_copies[index].passive_allocation_policy = tmp_passive_policy
        random_local_entropy_agent_copies[index].passive_allocation_policy = tmp_passive_policy

    ############################################################################
    # create tao_manager and add agent copies to it
    ############################################################################
        
    # initialise a TaoManager
    tao = TaoManager(env)
    out_dir = os.path.join(project_dir, 'outdir/target_graph/seed%d' % seed)
    max_num_core = 25
    tao.configs = {
        'max_num_core': max_num_core,
        'out_dir': out_dir,
        'initial_nodes': outbreak_data['initial_nodelist'],
        'surrogate_configs': surrogate_configs
        }
    
    # add agent copies to tao
    tao.add_agents(random_random_agent_copies)
    tao.add_agents(random_reactive_infected_agent_copies)
    tao.add_agents(random_node_entropy_agent_copies)
    tao.add_agents(random_bald_agent_copies)
    tao.add_agents(random_pagerank_agent_copies)
    tao.add_agents(random_degree_central_agent_copies)
    tao.add_agents(random_local_entropy_agent_copies)
    
    ############################################################################
    # add evaluator to tao_manager
    ############################################################################

    # initialise an evaluator, with AUC as designated metric
    evaluator_auc = BaseEvaluator('auc')
    # add evaluator to TaoManager
    tao.evaluator = evaluator_auc

    ############################################################################
    # run
    ############################################################################

    # run TAO (with parallelisation)
    tao.run(time_increment=1, up_to_time=len(outbreak_data['infection_states']), low_memory=True,
            save_infection_estimates=False, save_infection_point_estimates=True,
            with_hpdi=True, hpdi_ps=[0.9, 0.975, 0.95])
    
    # clear console
    os.system('clear')


# #### Print exit message

# In[4]:


# print exit time
from datetime import datetime
print('Exiting now (%s)...' % datetime.now().strftime("%H:%M:%S"))

