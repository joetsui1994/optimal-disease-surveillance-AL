import numpy as np

def simulate_binary_outbreak(graph, end_time, initial_infected_num=1, initial_infected_nodes=None, transmission_p=0.1):
    """Simulate a simple outbreak on a provided graph up to some finite time end_time."""
    # check if initial_infected_num is valid
    if initial_infected_num < 1 or initial_infected_num > len(graph.nodes()) or not isinstance(initial_infected_num, int):
        raise Exception('initial_infected_num must be a positive integer no greater than number of nodes in graph')
    # set initial infected nodes
    if not initial_infected_nodes:
        initial_infected_nodes = np.random.choice(graph.nodes(), initial_infected_num, replace=False)
    # construct infection_states
    infection_states = [1 if node in initial_infected_nodes else 0 for node in range(len(graph.nodes))]

    # simulate outbreak
    t = 0
    infection_states_history = {}
    while t != (end_time + 1):
        # get infected nodes
        infected_nodes = [node for node, infection_state in enumerate(infection_states) if infection_state == 1]
        for infected_node in infected_nodes:
            # get neighbors of infected node
            for neighbor in graph.neighbors(infected_node):
                # update infection_states probabilistically
                if infection_states[neighbor] == 0 and np.random.uniform(0, 1) < transmission_p:
                    infection_states[neighbor] = 1
        # store infection states
        infection_states_history[t] = infection_states.copy()
        # increment time
        t += 1

    return infection_states_history