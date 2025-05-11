# Modeling Scientific Theories as Causal Bayesian Networks

[Patrick Grim](https://pgrim.org/), Sophia Wushanley, Zhongming Jiang, Amber Campbell, Dennis Farmer

### Overview

This repository contains code developed for a research project on modeling scientific theories as causal Bayesian networks. We would like to encode a "theory of the world" as a network of statements with differing degrees of belief, with statements being connected by directed arrows to represent "conceptual support": `A->B` says that if there was high evidence that `A` was to be true, then `B` is likely true as well, and if `A` were to be disconfirmed, we would have less belief in `B`. This theory of the world would accept a stream of evidence from the real world, and use this evidence to adapt our theory to closer match the world.

`adaptive_bayesian_networks.py` contains algorithms for adapting network structures in response to evidence streams. We used simplified models with unweighted edges, since we represented the model's understanding of the world as a stream of binary evidence (see `adaptive_networks_given_evidence_stream.ipynb`). The current implementation would allow for probabilistic evidence streams given some modifications, although the real-world interpretation of such an evidence stream would still be just as difficult of a problem. 

### Potential Future Work

The format of the evidence plays a essential role in the methods used to adapt our theory to the world. We went on indefinite hiatus prior to the mainstream rise of AI chatbots; LLMs could serve as a potent evidence-generating mechanism for a similar adaptive causal Bayesian network model, and would potentially be a starting point for future work. 

The general idea of learning causal world models from an evidence stream (of some format) also has an application in the creation of effective AI agents:

Why Don't AI Agents Work? - Mutual Information: [link](https://www.youtube.com/watch?v=kpOWmwA6tJc)

Robust agents learn causal world models: [paper](https://arxiv.org/abs/2402.10877)

### Codebase

- **causal_discovery_functions.py**: Core utilities for:
  - Loading and saving causal discovery metrics and matrices
  - Simulating training data using Structural Equation Modeling (SEM)
  - Wrapping and evaluating causal discovery algorithms (PC, GES, NOTEARS, GOLEM) via the `Model` class
  - Plotting and comparing results to the ground truth
- **algorithms/**: Implementations of causal discovery methods:
  - `pc.py`: Peter-Clark (PC) algorithm (see `pc.gif`)
  - `ges.py`: Greedy Equivalence Search (GES) algorithm (see `ges.gif`)
- **adaptive_bayesian_networks.py**: Algorithms and data structures for
  - Representing Bayesian network structures (as binary strings, adjacency matrices)
  - Defining and iterating over evidence streams
  - Mutating and updating network structures given a temporal or event-based evidence stream
- **discover_causal_graphs.ipynb**: Jupyter notebook demonstrating causal discovery approaches
- **adaptive_networks_given_evidence_stream.ipynb**: Jupyter notebook illustrating performing genetic mutations on our theory of the world, given an oncoming stream of evidence.
- **gCastle Notes.py**: A writeup explaining the gCastle package and its use for causal discovery.

### Resources for more information
- Judea Pearl's *The Book of Why*
- "Review of Causal Discovery Methods Based on Graphical Models" by Glymour, Zhang, Spirtes [paper](/meeting_notes/Glymour%20Zhang%20Spirtes%20Review.pdf)
- [Brady Neal's Causal Inference Course](https://www.youtube.com/watch?v=CfzO4IEMVUk&list=PLoazKTcS0Rzb6bb9L508cyJ1z-U9iWkA0&index=1)
