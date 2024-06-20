from utils.helper_functions import generate_random_graph, load_graph, save_graph, load_infections_history, get_node_num, get_edge_num
from .outbreak_simulation import simulate_binary_outbreak
import numpy as np
import networkx as nx
import yaml
import os

class BinaryPrevalenceEnvironment:

    ############################ initialisations ############################

    def __init__(self, graph=None):
        self.graph = graph if graph is not None else generate_random_graph(10, 0.3)
        self.infections_history = None

    def initialise_static_infection_states(self, infection_states, up_time, start_time=0):
        """Initialises static infection states (from start_time, 0 by default) up to up_time."""
        # only proceed if infections_history is empty
        if self.infections_history:
            raise Exception('infections_history is not None, initialisation aborted')
        # check if start_time is valid
        if start_time < 0 or not isinstance(start_time, int):
            raise Exception('start_time must be a non-negative integer')
        # check if up_time is valid
        if up_time <= start_time or not isinstance(up_time, int):
            raise Exception('up_time must be a positive integer greater than start_time')
        # check if dimension of infection_states matches number of nodes in G
        if len(infection_states) != len(self.graph.nodes):
            raise Exception('Number of nodes in graph does not match dimension of infection states')
        # update infections_history
        infection_states_history = { time: infection_states for time in range(start_time, up_time+1) }
        self.infections_history = infection_states_history

    def initialise_simulated_infection_states(self, end_time, up_time=None, initial_infected_num=1, initial_infected_nodes=None, transmission_p=0.1, static=True):
        # only proceed if infections_history is empty
        if self.infections_history:
            raise Exception('infections_history is not None, initialisation aborted')
        # simulate outbreak
        infection_states = simulate_binary_outbreak(self.graph, end_time, initial_infected_num=initial_infected_num,
                                                    initial_infected_nodes=initial_infected_nodes, transmission_p=transmission_p)
        # update infections_history
        if static:
            end_infection_states = infection_states[end_time]
            # set up_time to end_time if not specified
            up_time = end_time if up_time is None else up_time
            self.initialise_static_infection_states(end_infection_states, up_time, start_time=0)
        else:
            self.infections_history = infection_states

    def set_infection_states(self, time=None, infection_states=None):
        # set time to maximum recorded time + 1 if not specified
        time = self.get_up_time() + 1 if time is None else time
        # set infection_states to most recent infection states if not specified (assuming static infection states)
        infection_states = self.get_infection_states() if infection_states is None else infection_states
        # check if time is valid
        if time < 0 or not isinstance(time, int):
            raise Exception('time must be a non-negative integer')
        # check if dimension of infection_states matches number of nodes in G
        if len(infection_states) != len(self.graph.nodes):
            raise Exception('Number of nodes in graph does not match dimension of infection states')
        # check if time already exists in infections_history
        if self.infections_history and time in self.infections_history:
            raise Exception('infections_states already exists for time {}'.format(time))
        # update infections_history
        if self.infections_history:
            self.infections_history[time] = infection_states
        else:
            self.infections_history = { time: infection_states }

    def set_infection_states_from_file(self, infections_history_infile, header=True):
        """Sets infection states from file."""
        # load infections_history from file
        infections_history = load_infections_history(infections_history_infile, header=header)
        # check if infections_history has the correct dimension
        if any(len(infection_states) != len(self.graph.nodes) for infection_states in infections_history.values()):
            raise Exception('Number of nodes in graph does not match dimension of infection states')
        # initialise infections_history
        self.infections_history = infections_history

    @classmethod
    def initialise_from_record(cls, graph_infile, infections_history_infile, header=True):
        """Initialises environment from record."""
        # load G from adjlist
        graph = load_graph(graph_infile)
        # convert node labels to integers
        node_mapping = { node: int(node) for node in graph.nodes() }
        graph = nx.relabel_nodes(graph, node_mapping)
        # load infections_history from file
        infections_history = load_infections_history(infections_history_infile, header=header)
        # check if infections_history has the correct dimension
        if any(len(infection_states) != len(graph.nodes()) for infection_states in infections_history.values()):
            raise Exception('Number of nodes in graph does not match dimension of infection states')
        # initialise environment
        env = cls(graph=graph)
        env.infections_history = infections_history
        
        return env

    ############################ book-keeping ############################

    def get_start_time(self):
        """Returns start_time of environment."""
        return min(self.infections_history.keys()) if self.infections_history else -1

    def get_up_time(self):
        """Returns up_time of environment."""
        return max(self.infections_history.keys()) if self.infections_history else -1

    ############################ I/O ############################

    def export_metadata(self, out_dir, filename=None):
        """Saves environment metadata to yaml file."""
        # check if out_dir exists
        os.makedirs(out_dir, exist_ok=True)
        output_filename = filename if filename else 'metadata.yaml'
        # create metadata dictionary
        metadata_out = {
            'node_num': get_node_num(self.graph),
            'edge_num': get_edge_num(self.graph),
            'start_time': self.get_start_time(),
            'up_time': self.get_up_time()
        }
        # write to file
        with open(os.path.join(out_dir, output_filename), 'w+') as outfile:
            yaml.dump(metadata_out, outfile, default_flow_style=False)

    def export_graph(self, out_dir, filename=None, format='adjlist'):
        """Saves graph to file."""
        filename = filename if filename else 'graph.%s' % format
        save_graph(self.graph, out_dir, filename, format=format)

    def export_infections_history(self, out_dir, filename=None, up_to_time=None):
        """Saves infections_history to file."""
        # set up_to_time to maximum recorded time if not specified
        up_to_time = self.get_up_time() if up_to_time is None else up_to_time
        # check if up_to_time is valid
        if up_to_time < 0 or up_to_time > self.get_up_time() or not isinstance(up_to_time, int):
            raise Exception('up_to_time must be a non-negative integer no greater than maximum recorded time')
        # save infections_history to file
        os.makedirs(out_dir, exist_ok=True) # check if out_dir exists, if not, create it
        output_filename = filename if filename else ('infections_history.time%d.csv' % up_to_time)
        with open(os.path.join(out_dir, output_filename), 'w+') as outfile:
            outfile.write('time,%s\n' % ','.join([str(node) for node in range(len(self.graph.nodes))]))
            # get sorted times
            sorted_times = sorted(self.infections_history.keys())
            # write infections_history from early to late
            outfile.write('\n'.join(
                ['%d,%s' % (time, ','.join([str(infection_state) for infection_state in self.infections_history[time]]))
                 for time in sorted_times if time <= up_to_time]))
                
    ############################ utilities ############################

    def get_infection_states(self, observed_nodes=None, time=None):
        """
        Returns infection states at time. If time is not specified, returns most recent infection states.
        If observed_nodes is specified, returns observed infection states. Otherwise, returns all infection states.
        """
        # set observed_nodes to all nodes if not specified
        observed_nodes = list(self.graph.nodes) if observed_nodes is None else observed_nodes
        # set time to maximum recorded time if not specified and cast to int
        time = int(min(self.get_up_time(), time) if isinstance(time, (int, float)) else self.get_up_time())
        # check if time exists in infections_history
        if time not in self.infections_history:
            raise Exception('infections_states does not exist for time {}'.format(time))
        
        # construct observed infection states
        observed_infection_states = [self.infections_history[time][node] if node in observed_nodes else -1 for node in range(len(self.graph.nodes))]
        return np.array(observed_infection_states)
        
    def get_last_infected_nodes(self, time=None):
        """Returns last infected nodes. If time is not specified, return most recently infected nodes."""
        # set time to maximum recorded time if not specified and cast to int
        time = int(min(self.get_up_time(), time) if isinstance(time, (int, float)) else self.get_up_time())
        # check if time exists in infections_history
        if time not in self.infections_history:
            raise Exception('infections_history does not exist for time {}'.format(time))
        # if time is 0, return all infected nodes
        if time == 0:
            infected_nodes = [node for node, infection_state in enumerate(self.infections_history[time]) if infection_state == 1]
        else:
            # get infections_history at time - 1 and time
            prev_infection_states = self.get_infection_states(time=time-1)
            curr_infection_states = self.get_infection_states(time=time)
            # compare prev_observed_infection_states and curr_infection_states to get most recent observations
            infected_nodes = [node for node, infection_state in enumerate(curr_infection_states) if infection_state == 1 and prev_infection_states[node] == 0]

        return infected_nodes