from .metrics import calculate_auc, calculate_pr_auc
import pandas as pd

class BaseEvaluator:

    ############################ class params ############################

    # allowed performance metrics
    DEFAULT_PERFORMANCE_METRICS = {
        'auc': calculate_auc,
        'pr_auc': calculate_pr_auc
    }

    ############################ initialisations ############################

    def __init__(self, metric):
        self.metric = metric
        self.custom_metrics = {}

        # check if metric is valid
        if self.metric not in self.DEFAULT_PERFORMANCE_METRICS:
            raise Exception('metric must be one of: %s' % ', '.join(self.DEFAULT_PERFORMANCE_METRICS.keys()))

    ############################ utilities ############################

    def set_metric(self, metric):
        """Sets metric."""
        # check if metric is valid
        if metric not in self.DEFAULT_PERFORMANCE_METRICS and metric not in self.custom_metrics:
            raise Exception('metric must be one of: %s' % ', '.join(list(self.DEFAULT_PERFORMANCE_METRICS.keys()) + list(self.custom_metrics.keys())))
        # set metric
        self.metric = metric

    def add_custom_metric(self, metric_name, metric_func, set_metric=True):
        """Adds custom metric."""
        # check that metric_name is a string
        if not isinstance(metric_name, str):
            raise Exception('metric_name must be a string')
        # check that metric_func is a function
        if not callable(metric_func):
            raise Exception('metric_func must be a function')
        # check if metric with the same name already exists
        if metric_name in self.DEFAULT_PERFORMANCE_METRICS or metric_name in self.custom_metrics:
            raise Exception('%s already exists as an available metric, please consider another name' % metric_name)
        # add custom metric
        self.custom_metrics[metric_name] = metric_func
        if set_metric:
            self.set_metric(metric_name)

    ############################ main ############################

    def evaluate_agent(self, agent, env, time=None):
        """
        Evaluates performance metrics for specified agents.
        If time is not specified, then performance is evaluated for all time-steps for which infection estimates are available.
        If time is specified, then performance is evaluated only for the specified time-step.
        """
        # get true infection_states_history with time-steps that match observed_infection_states_history
        time_matched_infection_states_history = { time: env.get_infection_states(time=time) for time in agent.surrogate.observed_infection_states_history.keys() }
        # construct kwargs, adding any additional arguments
        kwargs = {
            'time': time,
            'env_graph': env.graph,
            'infection_states_history': time_matched_infection_states_history,
            'observed_infection_states_history': agent.surrogate.observed_infection_states_history,
            'infection_estimates_history': agent.surrogate.infection_estimates_history
        }

        # evaluate performance
        if self.metric in self.custom_metrics:
            performance = self.custom_metrics[self.metric](**kwargs)
        else:
            performance = self.DEFAULT_PERFORMANCE_METRICS[self.metric](**kwargs)

        return performance

    def evaluate_multi_agents(self, agents, env, as_dataframe=True):
        """Evaluates performance of agents using specified metrics."""
        # evaluate performance for each agent
        agents_performance = { agent.id: self.evaluate_agent(agent, env) for agent in agents }
        # collect data for dataframe construction if specified
        if as_dataframe:
            df_data = [(agent.id,
                        agent.passive_allocation_policy.policy, agent.passive_allocation_policy.num_batch,
                        agent.allocation_policy.policy, agent.allocation_policy.num_batch,
                        agent.allocation_start_time, agent.query_rate,
                        time, len(agent.surrogate.get_unobserved_nodes(time=time)),
                        agent_performance) for agent in agents for time, agent_performance in agents_performance[agent.id].items()]
            df_columns = ['agent_id',
                          'passive_policy', 'passive_num_batch',
                          'active_policy', 'active_num_batch',
                          'active_start_time', 'query_time',
                          'time', 'num_unobserved_nodes', 'performance']
            df = pd.DataFrame(df_data, columns=df_columns)
            return df
        else:
            return agents_performance