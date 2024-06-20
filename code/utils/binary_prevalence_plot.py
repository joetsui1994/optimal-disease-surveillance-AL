import matplotlib.pyplot as plt
from IPython import display
import networkx as nx
import numpy as np
import imageio
import time
import os

def plot_observations(observed_infection_states, env_graph, pos, ax, desc='', highlighted_nodes=[], **kwargs):
    """Draws plot of either true infection states or observed infection states."""
    DEFAULT_PLOT_PARAMS = {
        'unobserved_node_size': 30,
        'unobserved_node_color': 'k',
        'unobserved_node_edge_color': 'k',
        'unobserved_node_edge_width': 0.1,
        'infected_node_size': 70,
        'infected_node_color': 'r',
        'infected_node_edge_color': 'r',
        'infected_node_edge_width': 0.1,
        'uninfected_node_size': 70,
        'uninfected_node_color': 'b',
        'uninfected_node_edge_color': 'b',
        'uninfected_node_edge_width': 0.1,
        'highlighted_node_edge_color': 'g',
        'highlighted_node_edge_width': 0.5,
        'alpha': 0.9,
        'edge_width': 0.5,
        'edge_color': 'k',
        'title_fontsize': 15,
    }
    # get custom params from kwargs if specified
    plot_params = { **DEFAULT_PLOT_PARAMS, **kwargs }

    # get node colours and sizes from observed_infection_states
    node_colors = { node: plot_params['unobserved_node_color'] if state == -1 else 
                   (plot_params['infected_node_color'] if state == 1 else plot_params['uninfected_node_color'])
                   for node, state in enumerate(observed_infection_states) }
    node_sizes = { node: plot_params['unobserved_node_size'] if state == -1 else
                  (plot_params['infected_node_size'] if state == 1 else plot_params['uninfected_node_size'])
                  for node, state in enumerate(observed_infection_states) }
    node_edge_colors = { node: plot_params['highlighted_node_edge_color'] if node in highlighted_nodes else
                        (plot_params['infected_node_edge_color'] if state == 1 else plot_params['uninfected_node_edge_color'])
                        for node, state in enumerate(observed_infection_states) }
    node_edge_widths = { node: plot_params['highlighted_node_edge_width'] if node in highlighted_nodes else
                        (plot_params['infected_node_edge_width'] if state == 1 else plot_params['uninfected_node_edge_width'])
                        for node, state in enumerate(observed_infection_states) }
    nx.draw(env_graph, pos, node_color=[node_colors[node] for node in env_graph.nodes], node_size=[node_sizes[node] for node in env_graph.nodes],
            alpha=plot_params['alpha'], width=plot_params['edge_width'], edge_color=plot_params['edge_color'],
            edgecolors=[node_edge_colors[node] for node in env_graph.nodes], linewidths=[node_edge_widths[node] for node in env_graph.nodes], ax=ax)
    # set title
    ax.set_title(desc, fontsize=plot_params['title_fontsize'])

def update_live_plot(observed_infection_states, env_graph, pos, fig, ax, hdisplay, desc='', **kwargs):
    """Updates live plot of either true infection states or observed infection states."""
    # clear previous plot
    ax.clear()
    # draw new plot
    plot_observations(observed_infection_states, env_graph, pos, ax, desc, **kwargs)
    # update display
    hdisplay.update(fig)
    # sleep for specified fps
    fps = kwargs.get('fps', 1)
    time.sleep(1/fps)

def animate_observations_history(env, agent=None, out_dir=None, gif_filename=None, **kwargs):
    """
    Animates history of obseravtions made by an agent. If no agent is specified,
    then only the infection states in the environment are shown (which might not be that interesting in the static case).
    """
    # generate node positions
    spring_k = kwargs.get('spring_k', 1.5)
    pos_seed = kwargs.get('pos_seed', np.random.randint(0, 2**16))
    pos = nx.spring_layout(env.graph, k=spring_k, seed=pos_seed)

    # print pos_seed generated if not provided, otherwise print pos_seed provided
    if 'pos_seed' not in kwargs:
        print('pos_seed generated: {}'.format(pos_seed))
    else:
        print('pos_seed provided: {}'.format(pos_seed))

    # if both out_dir and gif_filename are specified, then save gif of animation
    if out_dir is not None and gif_filename is not None:
        # check if out_dir exists
        os.makedirs(out_dir, exist_ok=True)
        fig_filenames = []
        # set up figure
        plot_dim = kwargs.get('plot_dim', (5, 3))
        fig, ax = plt.subplots(1, 1, figsize=plot_dim)
        # if agent is specified, then animate wrt to times of agent's observations
        if agent is not None:
            # iterate over observation times
            for time, observed_infection_states in agent.surrogate.observed_infection_states_history.items():
                # draw plot
                plot_observations(observed_infection_states, env.graph, pos, ax, desc='t={}'.format(time), **kwargs)
                # save figure
                fig_filename = os.path.join(out_dir, 't={}.png'.format(time))
                fig.savefig(fig_filename)
                fig_filenames.append(fig_filename)
                plt.close(fig)
        else: # otherwise, animate wrt to times in env
            # iterate over times in env
            for time in env.infections_history.keys():
                # draw plot
                plot_observations(env.get_infection_states(time=time), env.graph, pos, ax, desc='t={}'.format(time), **kwargs)
                # save figure
                fig_filename = os.path.join(out_dir, 't={}.png'.format(time))
                fig.savefig(fig_filename)
                fig_filenames.append(fig_filename)
                plt.close(fig)

        # generate gif
        figs = [imageio.imread(fig_filename) for fig_filename in fig_filenames]
        gif_fps = kwargs.get('gif_fps', 1)
        imageio.mimsave(os.path.join(out_dir, gif_filename), figs, fps=gif_fps)
        # remove figures
        for fig_filename in fig_filenames:
            os.remove(fig_filename)

    else: # otherwise, just show animation
        plot_dim = kwargs.get('plot_dim', (5, 3))
        fig, ax = plt.subplots(1, 1, figsize=plot_dim)
        hdisplay = display.display("", display_id=True)
        # if agent is specified, then animate wrt to times of agent's observations
        if agent is not None:
            for time, observed_infection_states in agent.surrogate.observed_infection_states_history.items():
                update_live_plot(observed_infection_states, env.graph, pos, fig, ax, hdisplay,
                                 desc='t={}'.format(time), **kwargs)
        else: # otherwise, animate wrt to times in env
            for time in env.infections_history.keys():
                update_live_plot(env.get_infection_states(time=time), env.graph, pos, fig, ax, hdisplay,
                                 desc='t={}'.format(time), **kwargs)
        # close plot
        plt.close(fig)
