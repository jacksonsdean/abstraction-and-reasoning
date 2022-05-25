import torch

from operations import all_operations

class Node():
    def __init__(self, activation):
        self.activation = activation
        self.sum_inputs = []
        self.current_output = []

class Connection():
    innovation = 0
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.innovation = Connection.innovation
        Connection.innovation += 1
        
    def __iter__(self):
        return iter([self.from_node, self.to_node])

def random_weight():
    return torch.rand(1)[0] * 2 - 1

def evaluate(input_sample, input_nodes, output_nodes, connections):
    """Evaluate the network to get data in parallel"""

    layers = feed_forward_layers(input_nodes, output_nodes, connections) # get layers
    for layer in layers:
        # iterate over layers
        for _, node in enumerate(layer):
            # iterate over nodes in layer
            # find incoming connections
            incoming = list(filter(lambda x, n=node: x.to_node == n, connections))
            # initialize the node's sum_inputs
            if layer == layers[0]:
                node.current_output = input_sample
            # else:
                # starting_input = torch.tensor([node.current_output])

            # initialize the sum_inputs for this node

            for cx in incoming:
                inputs = cx.from_node.current_output
                # inputs *= cx.weight # TODO
                node.current_output = inputs # TODO +=

            if len(incoming) > 0:
                # print(node.current_output)
                node.current_output = node.activation(node.current_output)  # apply activation
    
    # collect outputs from the last layer
    outputs = output_nodes[0].current_output
    # outputs = torch.tensor([node.current_output for node in output_nodes])
    return outputs


def get_candidate_nodes(s, connections):
    """Find candidate nodes c for the next layer.  These nodes should connect
    a node in s to a node not in s."""
    return set(b for (a, b) in connections if a in s and b not in s)

# Functions below are modified from other packages
# This is necessary because AWS Lambda has strict space limits,
# and we only need a few methods, not the entire packages.

###############################################################################################
# Functions below are from the NEAT-Python package https://github.com/CodeReclaimers/neat-python/

# LICENSE:
# Copyright (c) 2007-2011, cesar.gomes and mirrorballu2
# Copyright (c) 2015-2019, CodeReclaimers, LLC
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
# conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
# of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products
# derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
# OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################################


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.
    Returns a set of identifiers of required nodes.
    From: https://neat-python.readthedocs.io/en/latest/_modules/graphs.html
    """

    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return required


def feed_forward_layers(inputs, outputs, connections):
    """
    Collect the layers whose members can be evaluated in parallel in a feed-forward network.
    :param inputs: list of the network input nodes
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    Returns a list of layers, with each layer consisting of a set of node identifiers.
    Note that the returned layers do not contain nodes whose output is ultimately
    never used to compute the final network output.
    Modified from: https://neat-python.readthedocs.io/en/latest/_modules/graphs.html
    """
    required = required_for_output(inputs, outputs, connections)

    layers = []
    s = set(inputs)
    while 1:

        c = get_candidate_nodes(s, connections)
        # Keep only the used nodes whose entire input set is contained in s.
        t = set()
        for n in c:
            if n in required and all(a in s for (a, b) in connections if b == n):
                t.add(n)

        if not t:
            break

        layers.append(t)
        s = s.union(t)

    layers.insert(0, set(inputs)) # add input nodes as first layer

    return layers



if __name__ == "__main__":

    score = torch.zeros((len(fitness_functions)))
    
   
    
    import os
    import json
    from fitness import fitness_functions
    device = 'cpu'

    # test the network
    # create the network
    fct = all_operations[0]
    input_nodes = [Node(fct) for i in range(1)]
    output_nodes = [Node(fct) for i in range(1)]
    connections = [Connection(input_nodes[0], output_nodes[0], random_weight()),
                #    Connection(input_nodes[1], output_nodes[0], random_weight()),
    ]

    TRAIN_PATH = './ARC/data/training'
    training_tasks = sorted(os.listdir(TRAIN_PATH))
    task = training_tasks[0]
    task_path = os.path.join(TRAIN_PATH, task)
    with open(task_path, 'r') as f:
        task = json.load(f)
    task = task['train']
     # For each sample
    for sample in task:
        i = torch.tensor(sample['input']).to(device)
        o = torch.tensor(sample['output']).to(device)
        
        # For each fitness function
        for index, fitness_function in enumerate(fitness_functions):
            images = evaluate(i, input_nodes, output_nodes, connections)
            if images == []: # Penalize no prediction!
                score[index] += 500
            else: # Take only the score of the first output
                score[index] = fitness_function(images, o)
    print(tuple(score))