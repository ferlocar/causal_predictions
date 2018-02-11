from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import random
import pickle
import parser


# Class to represent a Bayesian Network
class BN:
    def __init__(self, size, iterations=100, magnitude=1):
        random.seed(0)
        np.random.seed(0)
        self.magnitude = magnitude
        # Add Utility and Treatment Effect nodes.
        size += 2
        self.size = size
        self.effect_ix = size - 2
        self.utility_ix = size - 1
        bn_f_name = 'data/bayesian_network.pkl'
        eq_f_name = 'data/equations.txt'
        # Causal structure equations
        self.equations = defaultdict(lambda: "0")
        if Path(bn_f_name).is_file() and Path(eq_f_name).is_file():
            with open(bn_f_name, 'rb') as f:
                obj = pickle.load(f)
                self.parents, self.edges = obj[0], obj[1]
            with open(eq_f_name, 'r') as f:
                eq_list = [parser.expr(line.rstrip('\n')).compile() for line in f]
                for i, eq in enumerate(eq_list):
                    self.equations[i] = eq
        else:
            # Generate Random BN
            self.edges = defaultdict(list)
            # Aux structure for undirected edges
            self.parents = defaultdict(list)
            # Initialize the BN edges (to make sure is fully connected). The Effect and Utility have no children.
            for i in range(size - 2):
                self.add_edge(i, i + 1)
            self.add_edge(size - 3, size - 1)
            # Do iterations to add/remove random edges
            for i in range(iterations):
                # Generate random pair of nodes
                u = np.random.randint(size)
                # Make sure utility and treatment effect don't have children (otherwise I'll have loop problems)
                if u >= self.effect_ix:
                    continue
                v = np.random.randint(size)
                # Remove edge if already in there. Otherwise add the edge.
                if v in self.edges[u]:
                    self.remove_edge(u, v)
                else:
                    self.add_edge(u, v)
            # Generate random conditional probabilities.
            with open(eq_f_name, 'w') as f:
                for v in range(self.size):
                    equation = "0"
                    parents = self.parents[v]
                    combinations = list(itertools.combinations(parents, 2)) + [(v,) for v in parents]
                    for combination in combinations:
                        equation += " + " + "*".join(["states[{}]".format(u) for u in combination])
                        if len(combination) == 1:
                            beta = np.random.uniform(-self.magnitude, self.magnitude)
                        else:
                            beta = np.random.choice([-1, 1]) * np.random.rand() ** 2
                        equation += "*{}".format(beta)
                    if v < self.effect_ix:
                        equation = "np.random.normal(loc=(" + equation + "))"
                    # Write equation to file
                    f.write(equation + '\n')
                    self.equations[v] = parser.expr(equation).compile()
            with open(bn_f_name, 'wb') as f:
                obj = [self.parents, self.edges]
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    # Attempts to remove an edge and returns if the operation was successful (the DAG remains connected).
    def remove_edge(self, u, v):
        self.edges[u].remove(v)
        self.parents[v].remove(u)
        connected = self.is_connected()
        if not connected:
            self.edges[u].append(v)
            self.parents[v].append(u)
        return connected

    # Attempts to add an edge and returns if the operation was successful (there are no cycles).
    def add_edge(self, u, v):
        self.edges[u].append(v)
        cycle = self.has_cycles()
        if cycle:
            self.edges[u].remove(v)
        else:
            self.parents[v].append(u)
        return not cycle

    # A recursive function used by has_cycles
    def has_cycles_aux(self, v, visited, finished):
        # Check if there's a cycle
        if visited[v] and not finished[v]:
            return True
        cycle = False
        # Mark the current node as visited.
        visited[v] = True
        # Go to all the vertices adjacent to this vertex
        for i in self.edges[v]:
            if not finished[i]:
                cycle = self.has_cycles_aux(i, visited, finished)
                if cycle:
                    break
        # Mark that we are done exploring this node
        finished[v] = True
        return cycle

    # Checks for cycles
    def has_cycles(self):
        # Mark all the vertices as not visited and not finished
        visited = [False] * self.size
        finished = [False] * self.size
        # Call the recursive helper function to check for cycles
        cycle = False
        for i in range(self.size):
            if not finished[i]:
                cycle = self.has_cycles_aux(i, visited, finished)
                if cycle:
                    break
        return cycle

    # A recursive function used by has_cycles
    def visit_aux(self, v, visited):
        # Mark the current node as visited.
        visited[v] = True
        # Go to all the vertices adjacent to this vertex
        for i in self.edges[v] + self.parents[v]:
            if not visited[i]:
                visited = self.visit_aux(i, visited)
        return visited

    # Checks if the graph is weakly connected
    def is_connected(self):
        # Mark all the vertices as not visited, and try to visit them all
        visited = [False] * self.size
        visited = self.visit_aux(0, visited)
        return all(visited)

    # Returns name of the vertex
    def get_v_name(self, v):
        if v == self.utility_ix:
            name = "U"
        elif v == self.effect_ix:
            name = "T"
        else:
            name = str(v)
        return name

    # Draw graph
    def draw_graph(self):
        g = nx.MultiDiGraph()
        all_edges = []
        for u in range(self.size):
            name_u = self.get_v_name(u)
            for v in self.edges[u]:
                name_v = self.get_v_name(v)
                all_edges.append((name_u, name_v))
        g.add_edges_from(all_edges)
        nx.draw(g, with_labels=True)
        plt.show()

    # Return the value of a node given the values of all the other nodes (states). States is used by self.equations.
    def eval_node(self, node_ix, states):
        return eval(self.equations[node_ix])

    # Generate a single sample using Gibbs Sampling
    def generate_sample(self, initial_states, iterations=1000):
        sample = list(initial_states)
        for ite in range(iterations):
            for v in range(len(sample)):
                sample[v] = self.eval_node(v, sample)
        return sample

    # Generate N samples using Gibbs Sampling
    def generate_samples(self, n):
        sample = np.zeros(self.size)
        samples = []
        for i in range(n):
            sample = self.generate_sample(sample)
            samples.append(sample)
        return samples


class Simulator(BN):
    def __init__(self, bn_size, observed_size, target_rate=0.50, avg_effect=0.05):
        super(Simulator, self).__init__(bn_size)
        # Add the treatment node
        self.size += 1
        # Index of the treatment in the sample (and in the Bayesian Network)
        self.treatment_ix = self.size - 1
        # Add simulation parameters
        self.base_rate = target_rate
        self.avg_effect = avg_effect
        # Only select variables with an index smaller than effect_ix (i.e., don't select utility or effect)
        self.observed = np.append(np.random.choice(self.effect_ix, observed_size, replace=False), [self.treatment_ix])
        # Index of the treatment in the data
        self.d_treatment_ix = len(self.observed) - 1
        self.min_u = 0
        self.fixed_effect = 0
        # Store sample details
        self.sample = None
        # Ixs for labels in sample
        self.label_t_ix = None
        self.label_u_ix = None

    # Return the labels that match the selected treatments
    def apply_treatments(self, treatments):
        outcomes = self.sample[:, self.label_u_ix] * (treatments - 1) + self.sample[:, self.label_t_ix] * treatments
        return outcomes.astype(dtype=bool)

    # Load sample from a File or generates Gibbs Sample
    def load_sample(self, f_name, size, noise_size_t=0.0, noise_size_u=0.0):
        if Path(f_name).is_file():
            sample = np.genfromtxt(f_name, delimiter=",")
            if len(sample) < size:
                extra_sample = np.array(self.generate_samples(size - len(sample)))
                sample = np.concatenate((sample, extra_sample), axis=0)
                np.savetxt(f_name, sample, delimiter=",")
            else:
                sample = sample[:size, :]
        else:
            sample = np.array(self.generate_samples(size))
            np.savetxt(f_name, sample, delimiter=",")
        np.random.shuffle(sample)
        # Estimate min_utility and fixed_effect
        utility = sample[:, self.utility_ix]
        self.min_u = np.percentile(utility, (1 - self.base_rate) * 100)
        effect = sample[:, self.effect_ix]
        self.fixed_effect = self.min_u - np.percentile(utility + effect, (1 - self.base_rate - self.avg_effect) * 100)
        sample[:, self.utility_ix] -= self.min_u
        sample[:, self.effect_ix] += self.fixed_effect
        # Add noise untreated
        labels_u = utility > 0
        noise_obs = np.random.binomial(1, noise_size_u, len(utility))
        new_labels = np.random.binomial(1, self.base_rate, len(utility))
        labels_u = labels_u * (1 - noise_obs) + noise_obs * new_labels
        # Add noise treated
        labels_t = utility + effect > 0
        noise_obs = np.random.binomial(1, noise_size_t, len(effect))
        new_labels = np.random.binomial(1, self.base_rate + self.avg_effect, len(effect))
        labels_t = labels_t * (1 - noise_obs) + noise_obs * new_labels
        # Add labels to the sample
        sample = np.column_stack((sample, labels_t, labels_u))
        self.label_t_ix = sample.shape[1] - 2
        self.label_u_ix = sample.shape[1] - 1
        self.sample = sample
        return sample
