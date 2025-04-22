

from pylab import *
import matplotlib.pyplot as plt
import os
import pyagrum as gum
import numpy as np
from string import ascii_lowercase as letters
import pyagrum.lib.notebook as get_backend
from IPython.display import display_html
from random import sample

# net: underlying pyAgrum.BayesNet object that is being wrapped
# adj: adjacency matrix representing the network
# bstr: binary string (1's and 0's) that represents the network
#       (represents skipping over the diagonal of the adjacency
#       matrix to make doing mutations easier)
#
# potential other attributes to consider adding:
# type: ??? temporal, probabilistic (classic Bayesian Network), causal, etc...
# TODO: add additional attributes to allow for
# conditional probability networks and temporal networks
# instead of purely deterministic causal networks


# net = BayesNet(gum.fastBN('a->b;a->c;b->d;b->e;c->e;c->f', 2))
#bstr = adj_to_bstr()

class bstr(str):
    def __init__(self, string: str = ""):
        self.b = string
        for c in self.b:
            if (c not in ["0", "1"]):
                raise(ValueError("bstr must contain only 0's and 1's"))
        # causes recursive cycle, likely not necessary
        #if not validate_network(self.b):
                #raise(ValueError("bstr must create a valid network, no cycles are allowed"))
        super(bstr, self).__init__()

    def __str__(self):
        return self.b


    def num_nodes(self) -> int:
        return int(np.sqrt(len(self.b) + len(self.b)/2))



class Evidence():
    def __init__(self, e: list[list[int]]):
        self.evidence = e
        self.initial_node = self.evidence[0][0]
    def __str__(self): return self.evidence.__str__()
    def __len__(self): return len(self.evidence)
    def __iter__(self): return (node_activation for node_activation in self.evidence)
    def __getitem__(self, key: int) -> 'list[int]': return self.evidence[key]
    def __setitem__(self, key, val): self.evidence[key] = val

class EvidenceList():
    """
    contains a *list* of activation lists / evidences, one 
    activation list / piece of evidence for each possible initial node activation in the network

    Given the following network: (assume directed downward)

    ```x
        a   f
       / \\ /
      b   c
       \\ / \\
        d <-e
    ```
    ```python
    evidence_list = [ 
    [ [a], [b,c], [d,e], [d] ],   # <- activate a: (explicit that 'a' happens), then
    [ [b], [d] ],                 #    b and c happen, then d and e, then d (again)
    [ [c], [d,e], [d] ],
    [ [d] ],
    [ [e], [d] ],
    [ [f], [c], [d,e], [d] ],
    ]
    ```
    Represented in the code as adj matrix indices (a=0, b=1, ...):

    ```python
    evidence_list = [ 
    [ [0], [1,2], [3,4], [3] ],   # <- activate a: (explicit that 'a' happens), then
    [ [1], [3] ],                 #    b and c happen, then d and e, then d (again) 
    [ [2], [3,4], [3] ],
    [ [3] ],
    [ [4], [3] ],
    [ [5], [2], [3,4], [3] ],
    ]
    ```
    """

    #                                           vvvv
    # if you use "__init__(self, e_l: list[...] = [])", it sets the value of 
    # [] globally whenever you construct an EvidenceList object, so when 
    # you try to run e = EvidenceList() you end up with whatever was 
    # constructed last time instead of an empty evidence (what?)
    def __init__(self, evidence_list: 'list[Evidence]'):
        self.evidence_list = evidence_list
        self.check_duplicates()
        self.num_nodes = len(self.evidence_list)
        self._index = -1

    def check_duplicates(self):
        # cannot store two (same or different) copies of the same initial activations
        activations = []
        for evidence in self.evidence_list:
            activations.append(evidence.initial_node)
        #print(activations)
        if len(activations) != len(set(activations)):
            raise ValueError("evidence_list contains duplicate initial activations")

    # overwrites the [] operator
    # locate the entry that encodes " 'key' happens " as the initial activation
    def __getitem__(self, key: int) -> 'Evidence':
        for evidence in self.evidence_list:
            if evidence.initial_node == key:
                return evidence
        return Evidence()

    def __setitem__(self, key, val):
        for i, evidence in enumerate(self.evidence_list):
            if evidence.initial_node == key:
                self.evidence_list[i] = val
                self.evidence_list[i].initial_node = self.evidence_list[i][0]
                return

    # allows iteration over the evidences in self.evidence_list (via for loops)
    # evidence_list = EvidenceList(...)
    # for e in evidence_list:
    #    .... do evidence stuff
    # returns a generator
    def __iter__(self):
        return (e for e in self.evidence_list)
    
    def __str__(self):
        s = ""
        if len(self.evidence_list) == 0:
            return "List is empty"
        for e in self.evidence_list:
            s += e.__str__()
            s += "\n"

        # replace numbers with letters for better readability
        # reverse letters so that 26 gets replaced before 2 and 6, etc.
        num_letter_pairs = {f"{n}": l for n, l in reversed(list(enumerate(letters)))}
        for key in num_letter_pairs.keys():
            s = s.replace(key, num_letter_pairs[key])

        return s

    def __repr__(self):
        return self.__str__()

    def append(self, val: Evidence):
        self.evidence_list.append(val)
        self.num_nodes = len(self.evidence_list)
        self.check_duplicates()

    def compare_to(self, variation: 'list[Evidence]') -> float:
        # for each activation list in the world (self), if it shows up at the same index
        # in the variation (if the list at the same index within the variation is 
        # identical to the one in the world)
        # then add one to the score
        score = 0
        world_size = 0

        for act_idx, activation_list in enumerate(self.evidence_list):
            if len(activation_list) == 0:
                world_size = world_size + 1
                if len(variation[act_idx]) == 0:
                    score += 1
            else:
                for happen_idx, node_happen in enumerate(activation_list):
                    world_size = world_size + 1
                    try:
                        if set(node_happen) == set(variation[act_idx][happen_idx]):
                            score += 1
                    except IndexError:
                        continue
                
        return score/world_size

    

def adj_to_evidencelist(adj: np.ndarray) -> EvidenceList:
    """
    create a collection of "event" tuples (using adj_traversal) that store the node that was 
    activated ("b happens", ...) along with the 'time' it was activated, then organize the events by 
    temporal order relative to each initial activation in an "evidence" object.
    """

    num_nodes = adj.shape[0]

    def adj_traversal(node_activations: 'list[int]', t: int = 1) -> 'set((int, int))':
        """
        allow for branching paths and collisions (either simutaneous or delayed) to be 
        accounted for within a network

        node_activations: a row of the adjacency matrix: `adj[activated_node_idx,:]`
        t: time relative to initial activation (keep as default in most cases)
        """
        events = set()
        for arc_idx, arc in enumerate(node_activations):
            if arc: # arc exists from node_idx to arc_idx (arc > 0)
                events.add((t, arc_idx))
                #events.add((t, arc_idx, arc)) # maybe "arc" later becomes the probability of a 
                                              # transition from node_idx to arc_idx?
                                              # (then I think the adjacency matrix would just be a transition
                                              # probability matrix of a markov chain with restrictions)
                                              # (topleft->bottomright diagonal = 0, no cycles such 
                                              # as 'a -> b -> c -> a', etc. etc.)
                events.union(adj_traversal(adj[arc_idx,:], t+1))
        return events

    all_evidence = []
    for _ in range(num_nodes):
        all_evidence.append([])

    for node_idx in range(num_nodes):
        curr_evidence = [[node_idx]]  # each piece of evidence stores whatever the initial activation is
        events = adj_traversal(adj[node_idx,:])
        if len(events) != 0:

            for _ in range(max(events)[0]):  # maximum value of t
                curr_evidence.append([])
            for t, arc_idx in events:
                curr_evidence[t].append(arc_idx)
        all_evidence[node_idx] = curr_evidence
    return EvidenceList([Evidence(e) for e in all_evidence])


class DirectedGraph():
    def __init__(self, _repr = 6):
        """
        initialize a DirectedGraph with either a pyAgrum.fastBN format string, 
        a binary string, an adjacency matrix, or with a pyAgrum.BayesNet object

        Parameters
        ----------
        _repr: pyAgrum.BayesNet, str, bstr, int, np.ndarray, or None
            object used to represent the initial BayesNet object
            
            if _repr is int, generate a network with _repr number of nodes
                (currently limited to 3, 4, or 6 nodes)


        fastBN format: uses semicolons (;) and arrows (a->b, b-a-c, etc)
            example: a->b;a->c;b->d;b->e;c->e;c->f
        """

        # do automatic conversion to bstr if possible
        if isinstance(_repr, str):
            if (";" not in _repr):
                try:
                    b = bstr(_repr)
                    _repr = b
                except:
                    pass



        if isinstance(_repr, list):
            if len(_repr) == 1:
                _repr = _repr[0]
            else:
                raise ValueError("BayesNet constructor recieved a list, but list is of length > 1 so automatic conversion to bstr/etc. is not possible\nrandom.sample with k=1 now returns a single element list so make sure to use sample(x, k=1)[0]")

        # Default network: tree structure with 6 nodes
        if _repr is None:
            # network is default pyagrum net
            self.net = gum.BayesNet()
            #binary string is empty
            self.b = bstr()
           #UPDATE: add evidence matrix and evidence string to BayesNet object
            self.ev_matr = np.empty([])
            self.ev_str = ""
            self.adj = np.empty([])
            # if there's a string representation, create network
        

        elif isinstance(_repr, int):
            if _repr == 3:
                self.net = gum.fastBN('a->b;b->c')
            if _repr == 4:
                self.net = gum.fastBN('a->b;a->c;b->d;c->d')
            if _repr == 6:
                self.net = gum.fastBN('a->b;a->c;b->d;b->e;c->e;c->f')


        # check if bstr before checking if str, since bstrs are considered as strings
        elif isinstance(_repr, bstr):
            arr = bstr_to_adj(_repr)
            self.net = adj_to_net(arr)
        elif isinstance(_repr, str):
            self.net = gum.fastBN(_repr)

        # if there's a matrix, create network
        elif isinstance(_repr, np.ndarray):
            self.net = adj_to_net(_repr)
            # if theres a network represntation, set it
        elif isinstance(_repr, gum.BayesNet):
            self.net = _repr
            # if we have some representation, make all of them using update
        #if _repr is not None:
            # if self.net was set above, update other attributes using internal self.net
        self.update()


    def show(self):
        """
        show network in a jupyter notebook
        """
        display_html(get_backend.getBN(self.net), raw=True)

    def __str__(self):
        output_str = ""
        num = self.adj.shape[0]
        nodes = list(map(chr, range(97, 97 + num)))
        for name in nodes:
            output_str += '    ' + str(name)
        output_str += '\n'
        for row in range(num):
            output_str += chr(97 + row) + '   '
            for col in range(num):
                val = int(self.adj[row][col])
                output_str += str(val) + '    '
            output_str += '\n'
        return output_str

    def update(self, _repr = None):
        """
        Update the network with a network representation
        if _repr is None, update using the internal self.net

        Parameters
        ----------
        _repr: pyAgrum.BayesNet, bstr, np.ndarray, or None
            object used to update the BayesNet object

        Examples
        --------

        >> bn = BayesNet("000000")
        >> bn.print()
        ""
        >> bn.update("100001")
        >> bn.print()
        "a->b, c->a"
        """
        

        #if we don't have something to add
        if _repr is None:
            # update based on current network object
            # set adjacency matrix based on network
            self.adj = net_to_adj(self.net)
            # set binary string based on adjacency matrix
            self.b = adj_to_bstr(self.adj)
            #UPDATE: updating the evidence portions too
            self.evidence = adj_to_evidencelist(self.adj)
            self.ev_matr = adj_to_ev_matr(self.adj)
            self.ev_str = ev_matr_to_estr(self.ev_matr)

        #given binary string
        elif isinstance(_repr, bstr):
            self.b = bstr(_repr)
            self.adj = bstr_to_adj(self.b)
            self.net = adj_to_net(self.adj)
            #UPDATE: updating evidence portions
            self.evidence = adj_to_evidencelist(self.adj)
            self.ev_matr = adj_to_ev_matr(self.adj)
            self.ev_str = ev_matr_to_estr(self.ev_matr)

        #given adjacency matrix

        elif isinstance(_repr, np.ndarray):
            self.adj = _repr
            self.b = adj_to_bstr(self.adj)
            self.net = adj_to_net(self.adj)
            #UPDATE: updating evidence portions
            self.evidence = adj_to_evidencelist(self.adj)
            self.ev_matr = adj_to_ev_matr(self.adj)
            self.ev_str = ev_matr_to_estr(self.ev_matr)

        #given network
        elif isinstance(_repr, gum.BayesNet):
            b = bstr(_repr)
            self.net = b
            self.adj = net_to_adj(self.net)
            self.b = adj_to_bstr(self.adj)
            #UPDATE: updating evidence portions
            self.evidence = adj_to_evidencelist(self.adj)
            self.ev_matr = adj_to_ev_matr(self.adj)
            self.ev_str = ev_matr_to_estr(self.ev_matr)
   
#just used to print the matrices nicely
#def pnet(adj):
    #num = adj.shape[0]
    #nodes = list(map(chr, range(97, 97 + num)))
    #for name in nodes:
        #print('   ',name, end='')
    #print('\n')
    #for row in range(num):
        #print(chr(97 + row), end='   ')
        #for col in range(num):
            #val = int(adj[row][col])
            #print(val, end = '    ')
        #print('\n')


def net_to_adj(net):
    # create a list of the nodes in the network
    nodes = list(net.nodes())
    # sort them (alphabetically?)
    nodes.sort()
    #create a 2d array to represent adjacency matrix
    #len(nodes) is the number of nodes, we want n x n adjacency matrix
    # fill them all in with zeroes (no arcs yet exist)
    adj = np.zeros((len(nodes), len(nodes)))
    # for each node, interate through the rest to see if there's an arc
    # between them, if there is, chance the adjacency matrix at those points
    # to be a 1
    for row in range(len(nodes)):
        for col in range(len(nodes)):
            if net.existsArc(nodes[row], nodes[col]):
                adj[row][col] = 1
    return adj
        
def adj_to_net(adj):
    # empty bayes net
    bn = gum.BayesNet("bn")
    # given a n x n array, we know there are n nodes in our network
    # get n by adj.shape[0]
    num_nodes = adj.shape[0]
    # letters = string.
    
    #CHANGED: creates node names starting with a, continuing alphabetically
    names = list(map(chr, range(97, 97 + num_nodes)))
    #names = np.char.array(['n'*num_nodes]) + np.arange(0,num_nodes).astype(str)
    nodes = [bn.add(name, 2) for name in names]
    for row in range(num_nodes):
        for col in range(num_nodes):
            if adj[row][col]:
                bn.addArc(nodes[row], nodes[col])
    return bn

def adj_to_bstr(adj: np.ndarray):
    num_nodes = adj.shape[0]
    output_str = ""
    for row in range(num_nodes):
        for col in range(num_nodes):
            if row == col:
                continue
            elif adj[row][col]:
                output_str += "1"
            else:
                output_str += "0"
    return bstr(output_str)


def bstr_to_adj(b: bstr) -> np.ndarray:
    num_nodes = b.num_nodes()
    adj = np.zeros((num_nodes, num_nodes))
    str_idx = 0
    for row in range(num_nodes):
        for col in range(num_nodes):
            if row == col:
                continue
            else:
                adj[row][col] = b[str_idx]
                str_idx += 1
    return adj


#UPDATE: function to go from adjacency matrix to evidence matrix
def adj_to_ev_matr(adj: np.ndarray):
    num_nodes = adj.shape[0]
    ev_matr = np.zeros((num_nodes, num_nodes))
    #copy over all the info from the adjacency matrix
    for n1 in range (num_nodes):
        for n2 in range(num_nodes):
            ev_matr[n1][n2] = adj[n1][n2]
    # for each node in our network
    #need to go through this whole process num_nodes number of times because we receive
    # new information with each iteration
    for i in range(num_nodes):
        for n1 in range (num_nodes):
            for n2 in range (num_nodes):
                for n3 in range (num_nodes):
                    #no updates to evidence if we are comparing the same node's information
                    if n1 == n2:
                        continue
                    #if we can get from node 1 to node 2 AND from node 2 to node 3
                    if ev_matr[n1][n2] > 0 and ev_matr[n2][n3] > 0:
                    #update the distance from node 1 to node 3 to be the distance from node 1 to node 2 +
                    #the distance from node 2 to node 3
                        ev_matr[n1][n3] = ev_matr[n1][n2] + ev_matr[n2][n3]      
                         
    return ev_matr 

#UPDATE: use this to retrieve a string from our evidence matrix
def ev_matr_to_estr(ev_matr):
    num_nodes = ev_matr.shape[0]
    e = ""
    for row in range(num_nodes):
        for col in range(num_nodes):
            if row == col:
                continue
            else: 
                e += str(int(ev_matr[row][col]))
    return e



#def compare_evidence(world: str, variation: str) -> int:
        #score = 0
        #for i in range(len(world)):
            #if world[i] == variation[i]:
                #score += 1
        #return score/len(world)

## UPDATE 11/17/22: Mutations


def flip_bit(s: bstr, i: int) -> bstr:
    """
    s: bstr consisting of 0's and 1's
    i: index of s to flip
    """
    return bstr(s[:i] + str(int(not bool(int(s[i])))) + s[i + 1:])

# Helper functions to make conversions less annoying:

def to_net(_repr):
    net = DirectedGraph(_repr)
    return net

def to_adj(_repr):
    net = DirectedGraph(_repr)
    return net.adj

def to_bstr(_repr):
    net = DirectedGraph(_repr)
    return net.b

# TODO: Potentially separate evidence matrix and adjacency matrix to two seperate classes
# so that isinstance(ev_mat, EvidenceMatrix), etc can be used

def show_nets(nets: "list[DirectedGraph]"):
    """
    Display networks side by side in a jupyter notebook

    see the below page for example, could add parameters to include labels:
        https://pyagrum.readthedocs.io/en/0.22.5/lib.notebook.html
    """
    if isinstance(nets, DirectedGraph):
        nets = [nets]
    get_backend.sideBySide(*[get_backend.getBN(n.net) for n in nets])

def single_mutations(b: bstr, frozen_arcs: 'list[int]' = []) -> 'list[bstr]':
    """
    Generate all valid single mutations for a network

    Parameters
    ----------
    b: bstr
        network representation to generate mutations from

    frozen_arcs: list[int]
            List of indices of the bstr network representation to
            exclude from changing when generating new mutations

    Examples
    --------

    >> single_mutations("100100")
    ['000100', '110100', '100000']

    >> single_mutations("100100", [0])
    ['100000', '110100']

    >> bn = BayesNet("000000")
    >> bn.print()
    ""
    >> bn.update("100001")
    >> bn.print()
    "a->b, c->a"

    """
    mutations = []
    num_nodes = b.num_nodes()
    for n, _ in enumerate(b):
        if n in frozen_arcs: continue
        new_net = flip_bit(b,n)
        if validate_network(new_net):
            mutations.append(new_net)
    return list(set(mutations))

def pairwise_mutations(b: bstr, frozen_arcs: 'list[int]' = []) -> 'list[bstr]':
    """
    Generate all valid pairwise mutations for a network

    Parameters
    ----------
    b: bstr
        network representation to generate mutations from

    frozen_arcs: list[int]
            List of indices of the bstr network representation to
            exclude from changing when generating new mutations
    """
    mutations = []
    num_nodes = b.num_nodes()
    for n, _ in enumerate(b):
        for m, _ in enumerate(b):
            if n==m: continue
            if (n in frozen_arcs) or (m in frozen_arcs): continue
            new_net = flip_bit(b,n)
            new_net = flip_bit(new_net,m)
            if validate_network(new_net):
                mutations.append(new_net)
    return list(set(mutations))

def validate_network(b: bstr):
    try:
        net = adj_to_net(bstr_to_adj(b))
    except gum.InvalidDirectedCycle:
        return False
    return True

# comparing two evidences:
def compare_evidence(world: EvidenceList, variation: EvidenceList) -> float:
    return world.compare_to(variation)
def compare_nets(world: DirectedGraph, variation: DirectedGraph) -> float:
    return compare_evidence(world.evidence, variation.evidence)

def compare_bstr(world: str, variation: str) -> int:
        score = 0
        for i in range(len(world)):
            if world[i] == variation[i]:
                score += 1
        return score/len(world)







# ----------------------------------------------------------------------------
# update 4/21/25: what in the world is going 
# on in o_gen/lo_gen/b_gen/lb_gen hahahaha 
# would help to structure the outputs of these a little
# better or collect some performance metrics and plot them


#will return the number of mutations needed to converge to correct model
#unguided 

        
# BASIC GEN (solely pairwise or solely single)
def b_gen(w: DirectedGraph, var: DirectedGraph, type: char) -> int:
    # if our variation and world are the same, zero mutations
    score = compare_evidence(w.evidence, var.evidence)
    if score == 1 and compare_bstr(w.b, var.b) == 1:
        return 0
    else:
        if type == 's':
            muts = single_mutations(var.b, [])
        elif type == 'p':
            muts = pairwise_mutations(var.b, [])
        highest = 0
        highest_str = 0
        for i in muts:
            curr = DirectedGraph(i)
            s = compare_evidence(w.evidence, curr.evidence)
            if s == 1 and compare_bstr(w.b, curr.b) == 1:
                return 1
            else:
                if s > highest:
                    highest = s
                    highest_str = i
        best_mut = highest_str
        best = DirectedGraph(best_mut)
        return 1 + b_gen(w, best, type) 

#  optimized gen: optimizes single vs double mutation depending on score
#if odd num wrong, do single, otherwise do pairwise
def o_gen(w: DirectedGraph, var: DirectedGraph) -> int:
    # if our variation and world are the same, zero mutations
    score = compare_evidence(w.evidence, var.evidence)
    if score == 1 and compare_bstr(w.b, var.b) == 1:
        return 0
    else:  
        correct = len(w.ev_str) * score
        if score == 1:
            correct = len(w.b) * compare_bstr(w.b, var.b)
        if correct % 2 == 1:
            muts = single_mutations(var.b, [])
        else:
            muts = pairwise_mutations(var.b, [])
        highest = 0
        highest_str = 0
        for i in muts:
            curr = DirectedGraph(i)
            s = compare_evidence(w.evidence, curr.evidence)
            if s == 1 and compare_bstr(w.b, curr.b) == 1:
                return 1
            else:
                if s > highest:
                    highest = s
                    highest_str = i
        best_mut = highest_str
        best = DirectedGraph(best_mut)
        return 1 + o_gen(w, best) 


# same as above functions but allows us to see the mutations and the current best variation


# loud optimized gen: optimizes single vs double mutation depending on score
#if odd num diff, do single, otherwise do pairwise
def lo_gen(w: DirectedGraph, var: DirectedGraph) -> int:

    wrong = 0 # wrong is defined!
    # if our variation and world are the same, zero mutations
    score = compare_evidence(w.evidence, var.evidence)
    if score == 1 and compare_bstr(w.b, var.b) == 1:
        return 0
    else:  
        correct = len(w.ev_str) * score
        if score == 1:
            correct = len(w.b) * compare_bstr(w.b, var.b)
        #if wrong % 2 == 1:
        if correct % 2 == 1:
            print("single")
            muts = single_mutations(var.b, [])
        else:
            print("pair")
            muts = pairwise_mutations(var.b, [])
        nets = [DirectedGraph(b) for b in sample(muts, len(muts))] 
        show_nets(nets)
        highest = 0
        highest_str = 0
        for i in muts:
            curr = DirectedGraph(i)
            s = compare_evidence(w.evidence, curr.evidence)
            if s == 1 and compare_bstr(w.b, curr.b) == 1:
                return 1
            else:
                if s > highest:
                    highest = s
                    highest_str = i
        best_mut = highest_str
        best = DirectedGraph(best_mut)
        best.show()
        return 1 + lo_gen(w, best) 

        # lb, LOUD, BASIC gen
def lb_gen(w: DirectedGraph, var: DirectedGraph, type: char) -> int:
    # if our variation and world are the same, zero mutations
    score = compare_evidence(w.evidence, var.evidence)
    if score == 1 and compare_bstr(w.b, var.b) == 1:
        return 0
    else:
        if type == 's':
            muts = single_mutations(var.b, [])
        elif type == 'p':
            muts = pairwise_mutations(var.b, [])
        nets = [DirectedGraph(b) for b in sample(muts, len(muts))] 
        show_nets(nets)
        highest = 0
        highest_str = 0
        for i in muts:
            curr = DirectedGraph(i)
            s = compare_evidence(w.evidence, curr.evidence)
            if score == 1 and compare_bstr(w.b, curr.b) == 1:
                return 1
            else:
                if s > highest:
                    highest = s
                    highest_str = i
        best_mut = highest_str
        best = DirectedGraph(best_mut)
        best.show()
        return 1 + lb_gen(w, best, type) 


# January 20th 1/20: Hybrid

def generate_hybrid(net1: DirectedGraph, net2: DirectedGraph, split: float = .5) -> DirectedGraph:
    #n = b1.num_nodes()
    b1 = net1.b
    b2 = net2.b
    n = len(b1.b)  # number of arcs
    arcs = list(range(n))
    loop_prevention = 10000
    final_bstr = ""
    result_is_valid = False

    while(not result_is_valid):
        loop_prevention = loop_prevention-1
        if loop_prevention == 0:
            print("loop limit exceeded")
            return bstr()
        final_bstr = ""
        s1_arc_idxs = sample(arcs, int(n*split))
        use_b1_arcs = list(np.zeros(n))
        for a in s1_arc_idxs:
            use_b1_arcs[a] = 1
        final_bstr = ""
        for i, b_choice in enumerate(use_b1_arcs):
            if b_choice == 1:
                final_bstr = final_bstr + b1[i]
            else:
                final_bstr = final_bstr + b2[i]
        final_bstr = bstr(final_bstr)
        result_is_valid = validate_network(final_bstr)
    return DirectedGraph(final_bstr)


def optimize_hybrid(world: DirectedGraph, v1: DirectedGraph, v2: DirectedGraph):

    v1_score = compare_nets(world, v1)
    v2_score = compare_nets(world, v2)
    if v1_score > v2_score:
        split = v1_score
    else:
        split = 1 - v2_score
    hybrid = generate_hybrid(v1, v2, split)
    hybrid_score = compare_nets(world, hybrid)
    print("scores for initial nets:")
    print("v1: ", round(v1_score,2))
    print("v2: ", round(v2_score,2))
    print("hybrid: ", round(hybrid_score,2))

    print("HYBRID SCORES:")
    #print("score split: v1_score / v2_score")
    for trial_n in range(1, 10+1):
        #v1_score = compare_nets(world, v1)
        #v2_score = compare_nets(world, v2)
        #if v1_score > v2_score:
            #split = v1_score
        #else:
            #split = 1 - v2_score
        #print("score split: ", round(v1_score, 2), " / ", round(v2_score, 2), sep=""))
        hybrid = generate_hybrid(v1, v2, split)
        hybrid_score = compare_nets(world, hybrid)
        print("trial ", trial_n, ": ", round(hybrid_score,2), sep="")


def optimize_single_and_hybrid(world: DirectedGraph, v1: DirectedGraph, v2: DirectedGraph):
    """
    see if hybrid net improves faster if it gets new information
    (v1 and v2 are changing by a single arc, so the v1/v2 split will likely
    change)
    """

    v1_score = compare_nets(world, v1)
    v2_score = compare_nets(world, v2)
    if v1_score > v2_score:
        split = v1_score
    else:
        split = 1 - v2_score
    hybrid = generate_hybrid(v1, v2, split)
    hybrid_score = compare_nets(world, hybrid)
    print("scores for initial nets:")
    print("v1: ", round(v1_score,2))
    print("v2: ", round(v2_score,2))
    print("hybrid: ", round(hybrid_score,2))

    print("HYBRID SCORES:")
    print("score split: v1_score / v2_score")
    for trial_n in range(1, 10+1):
        v1 = DirectedGraph(sample(single_mutations(v1.b), 1)[0])
        v2 = DirectedGraph(sample(single_mutations(v1.b), 1)[0])
        v1_score = compare_nets(world, v1)
        v2_score = compare_nets(world, v2)
        if v1_score > v2_score:
            split = v1_score
        else:
            split = 1 - v2_score
        print("score split: ", round(v1_score, 2), " / ", round(v2_score, 2), sep="")
        hybrid = generate_hybrid(v1, v2, split)
        hybrid_score = compare_nets(world, hybrid)
        print("trial ", trial_n, ": ", round(hybrid_score,2), sep="")








