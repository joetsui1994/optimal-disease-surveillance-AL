from pathos.multiprocessing import ProcessingPool as Pool
import coolname
import multiprocessing
from tqdm import tqdm
import math
import numpy as np
import copy
import yaml
import os
import json

def clear_console():
    # for Unix (Linux, macOS)
    _ = os.system('clear')

class TaoManager:

    ############################ initialisations ############################

    def __init__(self, env, name=None, agents=[], evaluator=None):
        self.name = name if name is not None else '_'.join(coolname.generate(2))
        self.env = env # only be one env
        self.agents = np.array(agents) # multiple agents
        self.evaluator = evaluator
        self.configs = None

    def reset_name(self, new_name=None):
        """Resets name of TaoManager. If new_name is not specified, generate new name from self."""
        self.name = new_name if new_name else '_'.join(coolname.generate(2))

    def parse_configs_yaml(self, yaml_file):
        """Parses configurations from YAML file."""
        # check if filepath exists
        if not os.path.exists(yaml_file):
            raise Exception('Specified file does not exist')
        # parse configurations from YAML file
        with open(yaml_file, 'r') as infile:
            configs = yaml.load(infile, Loader=yaml.FullLoader)
        # set configurations
        self.configs = configs

    ############################ environment ############################

    def get_infection_states(self, observed_nodes=None, time=None):
        """Returns infection states at time."""
        return self.env.get_infection_states(observed_nodes=observed_nodes, time=time)
    
    ############################ agents ############################

    def add_agents(self, agents):
        """Adds agents to TAO Manager."""
        self.agents = np.concatenate((self.agents, agents))

    def get_agent_by_id(self, agent_id):
        """Returns agent with specified agent_id."""
        # find agent with specified agent_id
        index_agent = [ index for index, agent in enumerate(self.agents) if agent.id == agent_id ]
        # return agent
        return self.agents[index_agent[0]] if len(index_agent) > 0 else None

    def get_agents_by_allocation_policy(self, allocation_policy, return_ids=False):
        """Returns a list of agents with specified allocation_policy."""
        # find agents with specified allocation_policy
        indices_agents = [ (index, agent) for index, agent in enumerate(self.agents) if agent.allocation_policy.policy == allocation_policy.lower() ]
        # return indices or agents
        return [index_agent[0 if return_ids else 1] for index_agent in indices_agents]
    
    def remove_agents_by_ids(self, agent_ids):
        """Removes agents with specified agent_ids."""
        # remove agents with specified agent_ids
        self.agents = np.delete(self.agents, agent_ids)

    ############################ allocation ############################

    def run(self, agent_ids=None, up_to_time=None, time_increment=0.1, **kwargs):
        """Main function to run TAO."""
        # if agent_ids is not specified, then get all agents (which have not been run)
        agents_to_run = [agent for agent in (self.agents if agent_ids is None else self.agents[agent_ids]) if not agent.is_run]
        # if up_to_time is not specified, then set up_to_time to env up_time
        up_to_time = min(self.env.get_up_time(), up_to_time) if isinstance(up_to_time, (int, float)) else self.env.get_up_time()
        # compute number of time-steps to run
        num_time_steps = math.ceil(up_to_time / time_increment)

        ############################

        # check if there are agents to run
        if len(agents_to_run) == 0:
            print('No agents to run, exiting now.')
            return

        # check if required params are specified in configs
        if self.configs is None or 'out_dir' not in self.configs:
            raise Exception('configs must contain out_dir if low_memory is enabled (forced)')

        # get passive and ative allocation-policy of all agents
        passive_allocation_policies = [agent.passive_allocation_policy.policy for agent in agents_to_run]
        active_allocation_policies = [agent.allocation_policy.policy for agent in agents_to_run]
        # compute summary statistics of passive and active allocation-policies
        passive_allocation_policies_summary = { policy: passive_allocation_policies.count(policy) for policy in set(passive_allocation_policies) }
        active_allocation_policies_summary = { policy: active_allocation_policies.count(policy) for policy in set(active_allocation_policies) }
        # print summary
        clear_console()
        print('Starting TAO-run for %s, up to %d steps at %f increment...' % (self.name, num_time_steps, time_increment))
        print('Allocation-policy summary:')
        print('  -> number of agents: %d' % len(agents_to_run))
        print('  -> (passive) allocation policies:\n%s' % '\n'.join(['        [%d] %s: %d' % (i + 1, policy, count) for i, (policy, count) in enumerate(passive_allocation_policies_summary.items())]))
        print('  -> (active) allocation policies:\n%s' % '\n'.join(['        [%d] %s: %d' % (i + 1, policy, count) for i, (policy, count) in enumerate(active_allocation_policies_summary.items())]))
        print('\n============================================')

        # low-memory mode is enabled (forced), and only point estimates will be saved
        with_hpdi = kwargs.get('with_hpdi', False)
        hpdi_ps = kwargs.get('hpdi_ps', [0.9, 0.95])
        print('Low-memory mode is enabled (forced), saving infection probability estimates [with_hpdi=%s,hpdi_ps=%s] at runtime to %s' % ('True' if with_hpdi else 'False',
                                                                                                                                          ','.join(str(hpdi_p) for hpdi_p in hpdi_ps),
                                                                                                                                          self.configs['out_dir']))

        ############################

        # TAO-loop to be run with for each agent in parallel
        def run_agent(args):

            # unpack args
            agent_i, agent = args
            out_dir = os.path.join(self.configs['out_dir'], self.name, agent.id) # get out_dir from configs

            # set up active-allocation schedule
            active_allocation_ticker = agent.allocation_start_time + 1 / agent.query_rate

            # clear newline generated by tqdm
            evaluation_save_first_line = True
            for time_i in tqdm(range(num_time_steps), position=agent_i, leave=False,
                               desc='[%s, %s(%d), %s(%d)]' % (agent.id, agent.passive_allocation_policy.policy, agent.passive_allocation_policy.num_batch, agent.allocation_policy.policy, agent.allocation_policy.num_batch)):
                
                # compute running time
                tao_time = time_i * time_increment

                # passive-allocation (fuzzy time-match)
                if tao_time <= agent.allocation_start_time < tao_time + time_increment:
                    # deploy passive allocation policy
                    selected_nodes = agent.deploy_passive_allocation_policy()
                    # get new observed infection states from environment
                    new_observed_infection_states = self.get_infection_states(observed_nodes=selected_nodes, time=tao_time)
                    # update surrogate with new observed infection states
                    agent.update_surrogate(tao_time, new_observed_infection_states)
                    
                # active-allocation
                elif tao_time <= active_allocation_ticker < tao_time + time_increment:
                    # deploy active-allcation policy
                    selected_nodes = agent.deploy_allocation_policy()
                    # get latest observed infection states from surrogate
                    last_observed_infection_states = agent.surrogate.get_observed_infection_states()
                    # get latest infection states of newly selected nodes
                    selected_nodes_infection_states = self.get_infection_states(observed_nodes=selected_nodes, time=tao_time)
                    # combine to get new observed infection states
                    new_observed_infection_states = [old_state if node not in selected_nodes else selected_nodes_infection_states[node] for node, old_state in enumerate(last_observed_infection_states)]
                    # update surrogate with new observed infection states
                    agent.update_surrogate(tao_time, new_observed_infection_states)
                    # update active_allocation_ticker
                    active_allocation_ticker += 1 / agent.query_rate

                    # evaluate agent performance at current time
                    if self.evaluator is not None:
                        self.evaluate_single_agent_performance(agent, time=tao_time, first_line=evaluation_save_first_line, save=True)
                        evaluation_save_first_line = False
                    # save infection probabilities
                    agent.surrogate.save_infection_estimates(out_dir, point_estimate_only=True, with_hpdi=with_hpdi, hpdi_ps=hpdi_ps)
                    # purge surrogate
                    agent.surrogate.purge_infection_estimates()

                # exit if there are no remaining unobserved nodes
                unobserved_nodes = agent.surrogate.get_unobserved_nodes()
                if (unobserved_nodes is not None and len(unobserved_nodes) == 0):
                    break

                # check if all unobserved nodes have the same infection state; if yes, break
                unobserved_infection_states = self.get_infection_states(observed_nodes=None, time=tao_time)[unobserved_nodes]
                if unobserved_nodes is not None and np.all(unobserved_infection_states == unobserved_infection_states[0]):
                    # populate remaining rows in metric file with None if evaluator is set
                    if self.evaluator is not None:
                        for time_i_2 in range(time_i+1, num_time_steps):
                            self.evaluate_single_agent_performance(agent, time=time_i_2 * time_increment, first_line=evaluation_save_first_line, save=True, save_none=True)
                    break

            # set run_state of agent to True
            agent.set_run()

            # return deepcopy of agent
            return copy.deepcopy(agent)

        ############################

        # get number of cores available on machine
        num_available_core = multiprocessing.cpu_count()
        # get number of cores to use from configs
        max_num_core = min(num_available_core, self.configs.get('max_num_core', num_available_core)) if self.configs is not None else num_available_core
        # print number of cores to use
        print('Number of cores to use: %d' % max_num_core)
        print('============================================\n')

        # save TAO-run metadata to file
        estimates_log_settings = {
            'with_hpdi': with_hpdi, 'hpdi_ps': hpdi_ps
        }
        self.save_metadata(agents_to_run, up_to_time, time_increment, estimates_log_settings)

        # update agents in parallel
        with Pool(processes=min(max_num_core, num_available_core)) as pool:
            updated_agents = pool.map(run_agent, [(agent_i, agent) for agent_i, agent in enumerate(agents_to_run)])

        # dictionary of agent-id and updated agent
        updated_agents = { agent.id: agent for agent in updated_agents }
        # update agents with updated_agents
        for agent_i, agent in enumerate(self.agents):
            self.agents[agent_i] = updated_agents[agent.id]

        # save observed infection states from surrogate to file
        for agent in updated_agents.values():
            # get out_dir from configs
            agent_out_dir = os.path.join(self.configs['out_dir'], self.name, agent.id)
            # save infection estimates
            agent.surrogate.save_observed_infection_states(agent_out_dir)
        
        # clear console before exiting
        clear_console()
        print('TAO-run completed, all output files saved to %s' % os.path.join(self.configs['out_dir'], self.name))
        print('============================================\n')
    
    ############################ evaluation ############################

    def evaluate_single_agent_performance(self, agent, time, first_line=False, save=False, save_none=False):
        """
        Evaluates performance of agent using specified metrics.
        Saves performance to out_dir if specified.
        If save_none is True, then save None to file; this is only applicable for when all unobserved nodes have the same infection state.
        """
        # exit if no evaluator has been added
        if self.evaluator is None:
            raise Exception('No valid evaluator has been added to TaoManager, please check')

        # evaluate performance for each agent
        agent_performance = { time: None } if save_none else self.evaluator.evaluate_agent(agent, self.env, time=time)
        
        # save perofrmance to file
        if save:
            # get out_dir from configs
            out_dir = os.path.join(self.configs['out_dir'], self.name, agent.id)
            # check if out_dir exists
            os.makedirs(out_dir, exist_ok=True)
            # save performance to file
            with open(os.path.join(out_dir, 'metric_%s.txt' % (self.evaluator.metric)), 'a') as outfile:
                if first_line:
                    outfile.write('time,%s\n' % self.evaluator.metric)
                outfile.write('%f,%s\n' % (time, ','.join([str(value) for value in agent_performance.values()])))
        else:
            return agent_performance

    ############################ I/O ############################

    def save_metadata(self, agents, up_to_time, time_increment, estimates_log_settings):
        """Save metadata relevant to TOA-run to a JSON file."""
        # check if out_dir exists
        if self.configs is None or 'out_dir' not in self.configs:
            raise Exception('configs must contain out_dir')
        # get out_dir from configs
        out_dir = os.path.join(self.configs['out_dir'], self.name)
        # check if out_dir exists
        os.makedirs(out_dir, exist_ok=True)
        # save metadata to file
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as outfile:
            metadata = {
                'name': self.name,
                'up_to_time': up_to_time,
                'time_increment': time_increment,
                'num_agents': len(agents),
                'agents': [{
                    'id': agent.id,
                    'passive_allocation_policy': agent.passive_allocation_policy.policy,
                    'passive_allocation_policy_num_batch': agent.passive_allocation_policy.num_batch,
                    'allocation_policy': agent.allocation_policy.policy,
                    'allocation_policy_num_batch': agent.allocation_policy.num_batch,
                    'allocation_start_time': agent.allocation_start_time,
                    'query_rate': agent.query_rate
                } for agent in agents],
                'configs': self.configs,
                'estimates_log_settings': estimates_log_settings
            }
            json.dump(metadata, outfile, indent=4)

    ############################ utilities ############################

    def reset_agents(self, agent_ids=None):
        """Resets agents with specified agent_ids."""
        # if agent_ids is not specified, then reset all agents
        agents_to_reset = self.agents if agent_ids is None else self.agents[agent_ids]
        # reset agents
        for agent in agents_to_reset:
            agent.reset_run()