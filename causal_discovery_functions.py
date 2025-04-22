import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch

import networkx as nx
import imageio
from pickle import load, dump
import os
import copy
from time import time
from unicodedata import normalize
from re import sub
from string import ascii_uppercase as letters

#import pyAgrum.lib.notebook as get_backend
import pyagrum.lib.notebook as get_backend

#from castle.common import GraphDAG
from castle.datasets import DAG, IIDSimulation, THPSimulation, Topology
from castle.algorithms import PC, GES, Notears, GOLEM, DirectLiNGAM, ICALiNGAM
from castle.metrics import MetricsDAG


def load_metrics(filename: str = "metrics.pkl") -> dict:
    filename = os.path.join(os.getcwd(), "training_data", filename)
    try:
        with open(filename, 'rb') as fp:
            d = load(fp)
        return d
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None

def save_metrics(d: dict, filename: str = "metrics.pkl"):
    filename = os.path.join(os.getcwd(), "training_data", filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as fp:
        dump(d,fp)

def load_causal_matrices(filename: str = "causal_matrices.pkl") -> dict:
    filename = os.path.join(os.getcwd(), "training_data/results", filename)
    try:
        with open(filename, 'rb') as fp:
            d = load(fp)
        return d
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None


def save_causal_matrices(d: dict, filename: str = "causal_matrices.pkl"):
    filename = os.path.join(os.getcwd(), "training_data/results", filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as fp:
        dump(d,fp)

class Model():
    # This class should make automating training of multiple
    # networks easier
    def __init__(self, name, algorithm: str, **kwargs):
        all_algorithms = ["pc", "ges", "notears", "golem"]

        self.algorithm = algorithm
        self.name = name
        self.parameters = {}
        self.train_time_sec = 0.0

        match self.algorithm:
            case "pc": 
                defaults = {"variant": "original", "alpha": 0.05, "ci_test": "fisherz", "priori_knowledge": None}
                self.parameters = {p: kwargs[p] if p in kwargs.keys() else defaults[p] for p in defaults.keys()}
                self.model = PC(variant=self.parameters["variant"],
                                alpha=self.parameters["alpha"],
                                ci_test=self.parameters["ci_test"],
                                priori_knowledge=self.parameters["priori_knowledge"])
            case "ges": 
                defaults = {"criterion": "bic", "method": "scatter", "k": 0.001, "N": 10}
                self.parameters = {p: kwargs[p] if p in kwargs.keys() else defaults[p] for p in defaults.keys()}
                self.model = GES(criterion=self.parameters["criterion"],
                                 method=self.parameters["scatter"],
                                 k=self.parameters["k"],
                                 N=self.parameters["N"])
            case "notears": 
                defaults = {"lambda1": 0.1, "loss_type": 'l2', "max_iter": 100, "h_tol": 1e-8, "rho_max": 1e+16, "w_threshold": 0.3}
                self.parameters = {p: kwargs[p] if p in kwargs.keys() else defaults[p] for p in defaults.keys()}
                self.model = Notears(lambda1=self.parameters["lambda1"],
                                     loss_type=self.parameters["loss_type"],
                                     max_iter=self.parameters["max_iter"],
                                     h_tol=self.parameters["h_tol"],
                                     rho_max=self.parameters["rho_max"],
                                     w_threshold=self.parameters["w_threshold"])
            case "golem": 
                defaults = {"B_init": None, "lambda_1": 2e-2, "lambda_2": 5.0, "equal_variances": True, "non_equal_variances": True,
                 "learning_rate": 1e-3, "num_iter": 1e+5, "checkpoint_iter": 5000, "seed": 1, "graph_thres": 0.3, "device_type": 'cpu',
                 "device_ids": 0}
                self.parameters = {p: kwargs[p] if p in kwargs.keys() else defaults[p] for p in defaults.keys()}
                self.model = GOLEM(B_init=self.parameters["B_init"], lambda_1=self.parameters["lambda_1"], lambda_2=self.parameters["lambda_2"], 
                                   equal_variances=self.parameters["equal_variances"], non_equal_variances=self.parameters["non_equal_variances"],
                                   learning_rate=self.parameters["learning_rate"], num_iter=self.parameters["num_iter"], 
                                   checkpoint_iter=self.parameters["checkpoint_iter"], seed=self.parameters["seed"],
                                   graph_thres=self.parameters["graph_thres"], device_type=self.parameters["device_type"],
                                   device_ids=self.parameters["device_ids"])

            case _: raise(f"{algorithm} not in {all_algorithms}")

    def learn(self, X, ground_truth):
        start_time = time()
        self.model.learn(X)
        self.train_time_sec = time() - start_time
        self.causal_matrix = self.model.causal_matrix
        self.metrics = MetricsDAG(B_est=self.causal_matrix, B_true = ground_truth).metrics
        self.metrics["n_undirected"] = get_n_undirected(self.causal_matrix != 0)

    def export(self, file=None):
        return (self.name, self.causal_matrix, self.metrics, self.algorithm, self.parameters, self.train_time_sec)

    def to_pkl(self):
        def slugify(value, allow_unicode=False):
            """
            Taken from https://github.com/django/django/blob/master/django/utils/text.py
            Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
            dashes to single dashes. Remove characters that aren't alphanumerics,
            underscores, or hyphens. Convert to lowercase. Also strip leading and
            trailing whitespace, dashes, and underscores.
            """
            value = str(value)
            if allow_unicode:
                value = normalize('NFKC', value)
            else:
                value = normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
            value = sub(r'[^\w\s-]', '', value.lower())
            return sub(r'[-\s]+', '-', value).strip('-_')

        d = {"name": self.name,
             "causal_matrix": self.causal_matrix, 
             "metrics": self.metrics, 
             "algorithm": self.algorithm, 
             "parameters": self.parameters, 
             "train_time_sec": self.train_time_sec
        }
        filename = os.path.join(os.getcwd(), "training_results", f"{slugify(d['name'])}_{d['algorithm']}.pkl")
    
    

def get_n_undirected(g):
    total = 0
    for i in range(g.shape[0]):
        for j in range(g.shape[0]):
            if (g[i, j] == 1) and (g[i, j] == g[j, i]):
                total += .5
    return total

def plot_dags(adj: 'list[np.matrix]', titles: 'list[str]' = None, shape: 'list[int]' = None, show=True, save_name=None, highlight_undirected=False, highlight_different_from_last = False):
    # AKA: GraphDAG but for multiple dags
    # adj: list of adjacency matrices

    adj = copy.deepcopy(adj)
    # trans diagonal element into 0
    #dag_1 = adj_1.copy()
    #dag_2 = adj_2.copy()
    #dag_3 = adj_3.copy()
    if not isinstance(adj, list):
        adj = [adj]
    n_adj = len(adj)
    if n_adj == 0: return
    n_nodes = len(adj[0])
    for dag_idx, dag in enumerate(adj):
        for i in range(n_nodes):
            if dag[i][i] == 1:
                adj[dag_idx][i][i] = 0


    if highlight_undirected:
        val = -1

        for dag_idx, dag in enumerate(adj):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if dag[i][j] == dag[j][i] and dag[i][j] != 0:
                        adj[dag_idx][i][j] = val
                        adj[dag_idx][j][i] = val

    if highlight_different_from_last and n_adj>1:
        val = 2
        epsilon = 1e-5
        ground_truth = adj[-1]
        for dag_idx, dag in enumerate(adj[:-1]):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if np.abs(dag[i][j] - ground_truth[i][j]) > epsilon and dag[i][j] != -1:
                        adj[dag_idx][i][j] = val
    
    if shape:
        nrows = shape[0]
        ncols = shape[1]
        if ncols > 3 or nrows < 1 or ncols < 1:
            shape = None
    if not shape:
        if n_adj <= 3:
            nrows = 1
            ncols = n_adj
        else:
            nrows = n_adj // 3
            if n_adj % 3 != 0:
                nrows = nrows + 1
            ncols = 3

    fig, axes = plt.subplots(figsize=(4*ncols, (4*nrows)-1), nrows=nrows, ncols=ncols)
    cmap = get_cmap("Greys")
    vmin = 0 - 1e-5
    vmax = 1 + 1e-5
    # verify that this is indeed true
    handles = [Patch(color="black", label = "Arc Exists\n(Y->X)")]

    cmap.set_under("g")
    if highlight_undirected:
        handles.append(Patch(color="g", label="Undirected\n(X<=>Y)"))

    cmap.set_over("r")
    if highlight_different_from_last:
        handles.append(Patch(color="r", label="Not in\nGround Truth"))

    if titles is None:
        titles = [f"dag_{i+1}" for i in range(n_adj-1)]
        titles.append("ground_truth")


    if n_adj == 1:
        ax_iter = [axes]
    elif nrows == 1:
        ax_iter = axes
    else:
        ax_iter = axes.ravel()
    
    for idx, ax in enumerate(ax_iter):
        # accounts for possible difference
        # between n_adj and nrow*ncol
        if idx == n_adj:
            break
        ax.set_title(titles[idx])
        map = ax.imshow(adj[idx], cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
        #fig.colorbar(map, ax=ax)
        ax.grid()
        ax.set_xticks(np.arange(n_nodes))
        ax.set_xticklabels(list(letters[:n_nodes]))
        ax.set_yticks(np.arange(n_nodes))
        ax.set_yticklabels(list(letters[:n_nodes]))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc="center left", fancybox=True, shadow=True, handles=handles, bbox_to_anchor=(1, 0.5))

    if save_name is not None:
        save_name = os.path.join(os.getcwd(), "training_data", save_name)
        fig.savefig(save_name)
    if show:
        plt.show()

def show_nets(nets: 'list[np.ndarray]'):
    converted_nets = []
    if isinstance(nets, np.ndarray):
        nets = [nets]
    for adj in nets:
        G = nx.DiGraph()
        n_nodes = adj.shape[1]
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj[i][j] == 1:
                    G.add_edge(i,j)
        G_pydot = nx.nx_pydot.to_pydot(G)
        #extract pos from nodes
        #pydot.Node().get_attributes()
        #print(G_pydot.nodes())

        converted_nets.append(G_pydot)

    get_backend.sideBySide(*[get_backend.getGraph(n) for n in converted_nets])

    # california's kern river and the 2023 snow melt

def show_nets_circular(adj, node_labels=None, save_name = None, show=False, graph_label=""):
    g = nx.DiGraph(adj)
    if node_labels is not None and adj.shape[0] == len(node_labels):
        MAPPING = {k: v for k, v in enumerate(node_labels)}
        g = nx.relabel_nodes(g, MAPPING, copy=True)
    #plt.figure(figsize=(6, 4))
    fig, ax = plt.subplots(1,1, figsize=(6,4))
    nx.draw(
        G=g,
        ax=ax,
        with_labels = True,
        #node_color = "green",
        node_color = "black",
        #font_color="black",
        font_color="white",
        font_size=int(10*3),
        node_size=int(600*3),
        arrowsize = 35,
        pos=nx.circular_layout(g)
    )
    ax.set_title(graph_label)
    if save_name is not None:
        #save_name = os.path.join(os.getcwd(), save_name)
        plt.savefig(save_name)
    if show:
        plt.show()


def _save_progress_frames(progress_frames: list, node_labels = None, name = "results"):
    #if node_labels is None:
        #if len(progress_frame
        #node_labels = 

    filename = os.path.join(os.getcwd(), "training_data", "progress_frames.pkl")
    with open(filename, 'wb') as fp:
        dump(progress_frames,fp)



    os.makedirs(os.path.join(os.getcwd(), "progress_frames", name), exist_ok=True)

    i = 0
    for graph, label, info in progress_frames:

        i = i+1
        g = nx.DiGraph(graph)
        if node_labels is not None and progress_frames[0][0].shape[0] == len(node_labels):
            MAPPING = {k: v for k, v in enumerate(node_labels)}
            g = nx.relabel_nodes(g, MAPPING, copy=True)
        #plt.figure(figsize=(6, 4))
        fig, ax = plt.subplots(figsize=(6,4))
        nx.draw(
            G=g,
            ax=ax,
            with_labels = True,
            #node_color = "green",
            node_color = "black",
            #font_color="black",
            font_color="white",
            font_size=int(10*3),
            node_size=int(600*3),
            arrowsize = 35,
            pos=nx.circular_layout(g)
        )
        ax.set_title(f"{label}, {info}")
        #plt.subplots_adjust(hspace=3)
        #plt.figtext(0.5, 0.01, , wrap=True, horizontalalignment='center', fontsize=12)

        save_name = os.path.join("progress_frames", name, f"graph_{str(i).zfill(4)}.png")
        fig.savefig(save_name)
        plt.close()

def progress_frames_to_gif(progress_frames: list, node_labels = None, name = "results", frame_duration = 1000, delete_frame_pngs=True, gif_name = None):
    """
    Convert a list of progress frames to a gif.
    """
    if gif_name is None:
        gif_name = f"{name}.gif"

    _save_progress_frames(progress_frames, node_labels, name)

    # Create a gif from the images
    # duplicating the first and last image

    images = []
    img_path = ""
    for _ in range(2):
        img_path = os.path.join("progress_frames", name, f"graph_{str(1).zfill(4)}.png")
        images.append(imageio.imread(img_path))
    for i in range(1, len(progress_frames)+1):
        img_path = os.path.join("progress_frames", name, f"graph_{str(i).zfill(4)}.png")
        images.append(imageio.imread(img_path))
    for _ in range(4):
        images.append(imageio.imread(img_path))
    
    gif_path = os.path.join("progress_frames", name, gif_name)
    imageio.mimsave(gif_path, images, duration=frame_duration)
    if delete_frame_pngs:
        for i in range(1, len(progress_frames)+1):
            img_path = os.path.join("progress_frames", name, f"graph_{str(i).zfill(4)}.png")
            os.remove(img_path)

# simulation data
def simulate_iid(method = 'linear', sem_type = 'gauss', n_nodes = 10, n_edges = 15, n = 2000):
    # method: structural assignment
    # n: number of observations
    weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges, weight_range=(0.5, 2.0), seed=1)
    dataset = IIDSimulation(W=weighted_random_dag, n=n, method=method, sem_type=sem_type)
    true_dag, X = dataset.B, dataset.X
    return true_dag, X


def generate_thp(method = 'linear', sem_type = 'gauss', n_nodes = 10, n_edges = 15, n = 2000):
        """
        A class for simulating event sequences with
        THP (Topological Hawkes Process) setting.

        Parameters
        ----------
        causal_matrix: np.matrix
            The casual matrix.
        topology_matrix: np.matrix
            Interpreted as an adjacency matrix to generate graph.
            Has two dimension, should be square.
        mu_range: tuple, default=(0.00005, 0.0001)
        alpha_range: tuple, default=(0.005, 0.007)
        """
        true_graph_matrix = DAG.erdos_renyi(n_nodes=10, n_edges=10)
        topology_matrix = Topology.erdos_renyi(n_nodes=20, n_edges=20)
        simulator = THPSimulation(causal_matrix = true_graph_matrix, 
                                  topology_matrix = topology_matrix,
                                  mu_range=(0.00005, 0.0001),
                                  alpha_range=(0.005, 0.007))
        data = simulator.simulate(T=25000, max_hop=2)
        return {"dag": true_graph_matrix, "X": data, "topology_matrix": topology_matrix}