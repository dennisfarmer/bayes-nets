import os
os.environ['CASTLE_BACKEND'] = 'pytorch'
#from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation, load_dataset
from castle.algorithms import PC, GES, Notears, GOLEM
from castle.common.priori_knowledge import PrioriKnowledge

import networkx as nx
import pandas as pd
from numpy.random import normal
import numpy as np

from string import ascii_uppercase as letters
from time import time
#from pgrim.discovery import *
from functions import *


def training_example():

    #data = generate_iid(n_nodes=6, n_edges=10)
    # ground truth: adjacency matrix with 0's and 1's (np.matrix)
    #ground_truth = data["dag"]
    #print(ground_truth)
    N = 1000
    dag = nx.DiGraph()
    dag.add_nodes_from(list(letters[:6]))
    dag.add_edges_from([("B", "A"), ("F", "A"), ("A", "C"), ("B", "C"), ("D", "C"), ("B", "D"), ("B", "E"), ("D", "E"), ("B", "F")])
    ground_truth = nx.to_numpy_array(dag).astype(int)

    U = {"A": normal(1,1.2,N), "C": normal(0.2, 0.4, N), "D": normal(size=N), "E": normal(1.3, 0.4, N), "F": normal(0.7, 2, N)}
    B = normal(size=N)
    F = 0.7*B + U["F"]
    A = 0.8*B + F + U["A"]
    D = B + U["D"]
    C = A + B + 0.7*D + U["C"]
    E = B + 0.9*D + U["E"]
    del U

    X = np.vstack([A, B, C, D, E, F]).T
    X.tofile("training_data/causal_matrix.csv", sep=",")
    priori = PrioriKnowledge(6)
    priori.add_required_edge(1, 3)  # we know that connection exists from B to D (for some reason)
    pc = Model("SCM_Demo", "pc", variant = "original", priori_knowledge=priori)
    pc.learn(X, ground_truth)
    causal_matrix = pc.causal_matrix

    show_nets_circular(ground_truth, node_labels = list(letters[:6]), save_name="ground_truth.png", show=False)
    show_nets_circular(causal_matrix, node_labels = list(letters[:6]), save_name="causal_matrix.png", show=False)
    plot_dags([causal_matrix, ground_truth], highlight_undirected=True, highlight_different_from_last=False, show=False, save_name="dag.png")
    pc.to_pkl()

def main():
    training_example()
    print("Done!")

    



if __name__ == "__main__":
    main()
