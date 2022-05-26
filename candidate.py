import copy
import random
import numpy as np
import torch
from fitness import fitness_functions
from operations import all_operations

from util import plot_one, show_image_list

prob_add_connection = 0.5
prob_add_node = 0.5
prob_remove_node = 0.3
prob_remove_connection = 0.3
prob_mutate_weight = 0.3
prob_mutate_activation = 0.3

initial_connection_prob = 1.0
initial_hidden = 0

max_weight = 1.0
weight_mutation = .2

device = 'cpu'

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

class Candidate():
    def __init__(self, num_inputs=1, num_outputs=1) -> None:
        self.input_nodes = []
        self.output_nodes = []
        self.connections = []
        self.hidden_nodes = []
        
        for _ in range(num_inputs):
            self.input_nodes.append(Node(random_activation()))
        for _ in range(num_outputs):
            self.output_nodes.append(Node(random_activation()))
        for _ in range(initial_hidden):
            self.hidden_nodes.append(Node(random_activation()))

        for inp in self.input_nodes:
            for h in self.hidden_nodes:
                if random.random() < initial_connection_prob:
                    self.connections.append(Connection(inp, h, random_weight()))
        if initial_hidden>0:
            for h in self.hidden_nodes:
                for outp in self.output_nodes:
                    if random.random() < initial_connection_prob:
                        self.connections.append(Connection(h, outp, random_weight()))
        else:
            for inp in self.input_nodes:
                for outp in self.output_nodes:
                    if random.random() < initial_connection_prob:
                        self.connections.append(Connection(inp, outp, random_weight()))
    def layers(self):
        return feed_forward_layers(self.input_nodes, self.output_nodes, self.connections)

    def add_node(self):
        if len(self.hidden_nodes) == 0 or len(self.connections) == 0:
            self.hidden_nodes.append(Node(random_activation()))
            return
        # pick random connection to break
        cx = random.choice(self.connections)
        # create new node
        new_node = Node(random_activation())
        self.hidden_nodes.append(new_node)
        # create new connections
        new_connection_a = Connection(cx.from_node, new_node, 1.0)
        new_connection_b = Connection(new_node, cx.to_node, random_weight())
        # replace old connection with new connections
        self.connections.append(new_connection_a)
        self.connections.append(new_connection_b)
        self.connections.remove(cx)

    def remove_node(self):
        if len(self.hidden_nodes) == 0:
            return
        random_node = random.choice(self.hidden_nodes)
        for cx in self.connections[::-1]:
            if cx.from_node == random_node or cx.to_node == random_node:
                self.connections.remove(cx)

    def add_connection(self):
        ffl = self.layers()
        for i, l in enumerate(ffl[::-1]):
            if len(l)==0:
                ffl.remove(l)
            ffl[i] = list(set(l)) 
            
        start_layer = random.randint(0, len(ffl)-1)
        start_layer = min(start_layer, len(ffl)-2)
        start_node = random.randint(0, len(ffl[start_layer])-1)
        end_layer = random.randint(start_layer+1, len(ffl)-1)
        end_node = random.randint(0, len(ffl[end_layer])-1)
        self.connections.append(Connection(list(ffl[start_layer])[start_node], ffl[end_layer][end_node], random_weight()))

    def remove_connection(self):
        if len(self.connections) == 0:
            return
        cx = random.choice(self.connections)
        self.connections.remove(cx)

    def mutate_activations(self):
        for node in self.hidden_nodes + self.output_nodes + self.input_nodes:
            if random.random() < prob_mutate_activation:
                node.activation = random_activation()
        
    def mutate_weights(self):
        for cx in self.connections:
            if random.random() < prob_mutate_weight:
                cx.weight += random.gauss(0, weight_mutation)
                if cx.weight < -max_weight:
                    cx.weight = max_weight
                if cx.weight > max_weight:
                    cx.weight = max_weight

    def mutate(self):
        if random.random() < prob_add_node:
            self.add_node()
        if random.random() < prob_add_connection:
            self.add_connection()
        if random.random() < prob_remove_node:
            self.remove_node()
        if random.random() < prob_remove_connection:
            self.remove_connection()
        self.mutate_weights()
        self.mutate_activations()

    
    def evaluate(self, input_sample):
        """Evaluate the network to get data in parallel"""

        layers = feed_forward_layers(self.input_nodes, self.output_nodes, self.connections) # get layers
        for layer_index, layer in enumerate(layers):
            # iterate over layers
            for _, node in enumerate(layer):
                # iterate over nodes in layer
                # find incoming connections
                incoming = list(filter(lambda x, n=node: x.to_node == n, self.connections))
                # initialize the node's sum_inputs

                # initialize the sum_inputs for this node
                if layer_index == 0:
                    node.current_output = [input_sample]
                else:
                    node.current_output = [] # reinitialize the node's current_output

                for cx in incoming:
                    from_layer_index = 0
                    for l in layers:
                        if cx.from_node in l:
                            break
                        from_layer_index += 1
                    to_layer_index = 0
                    for l in layers:
                        if cx.to_node in l:
                            break
                        to_layer_index += 1
                    if from_layer_index == to_layer_index:
                        continue
                    if from_layer_index > to_layer_index:
                        continue
                    inputs = cx.from_node.current_output
                    # print(inputs)
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].type(torch.float32)
                        inputs[i] *= float(cx.weight) 
                        
                        if i >= len(node.current_output):
                            node.current_output.append(inputs[i])
                        else:
                            node.current_output[i], inputs[i] = pad_to_same(node.current_output[i], inputs[i])
                            node.current_output[i] = torch.add(node.current_output[i], inputs[i])
                        # print(i, len(node.current_output), len(inputs))
                # node.current_output = [x for x in node.current_output]
                # print(node.current_output)
                node.current_output = node.activation(node.current_output)  # apply activation
                node.current_output = [torch.nan_to_num(x) for x in node.current_output]

        # collect outputs from the last layer
        outputs = [o.current_output for o in self.output_nodes]
        # outputs = torch.tensor([node.current_output for node in output_nodes])
        return outputs

    def evaluate_fitness(self, task):
        score = torch.zeros((len(fitness_functions)))
        for sample in task:
            i = torch.tensor(sample['input']).to(device)
            o = torch.tensor(sample['output']).to(device)
            i = i.type(torch.FloatTensor)
            o = o.type(torch.FloatTensor)

            # For each fitness function
            for index, fitness_function in enumerate(fitness_functions):
                images = self.evaluate(copy.deepcopy(i))
                if images == []: # Penalize no prediction!
                    score[index] += 500
                else: # Take only the score of the first output
                    for img in range(len(images)):
                        if not isinstance(images[img], torch.Tensor):
                            if len(images[img])==2:
                                images[img][0], images[img][1] = pad_to_same(images[img][0], images[img][1])
                                images[img] = torch.stack(images[img])
                            else:
                                try:
                                    images[img] = images[img][0]
                                except:
                                    images[img] = torch.tensor(images[img])
                                    continue
                    images = [img for img in images if img.shape[0] > 0 and img.shape[1] > 0]
                    if len(images) == 0 or images == [] or len(images[0].shape)<2:
                        score[index] += 500
                        continue
                    # remove dimensions of size 1
                    images = [img.squeeze() for img in images]

                    if len(images[0].shape) == 3:
                        images[0] = images[0][0]
                    images[0] = torch.nan_to_num(images[0], 0)
                    images[0] = images[0].type(torch.float32)
                    images[0] /= torch.max(images[0])
                    images[0] *=9.0
                    torch.round(images[0])
                    score[index] = fitness_function(images[0], o)
        return score
    
def random_weight():
    return random.uniform(-max_weight,max_weight)
def random_activation():
    return random.choice(all_operations)
def get_candidate_nodes(s, connections):
    """Find candidate nodes c for the next layer.  These nodes should connect
    a node in s to a node not in s."""
    return set(b for (a, b) in connections if a in s and b not in s)

def pad_to_same(x, y):
    # pad the output to match the input
    if x.shape[-2] < y.shape[-2]:
        x = torch.nn.functional.pad(x, (0,0, y.shape[-2] - x.shape[-2], 0), value=-1)
    elif x.shape[-2] > y.shape[-2]:
        y = torch.nn.functional.pad(y, (0,0, x.shape[-2] - y.shape[-2], 0), value=-1)
    if x.shape[-1] < y.shape[-1]:
        x = torch.nn.functional.pad(x, (0, y.shape[-1] - x.shape[-1], 0, 0), value=-1)
    elif x.shape[-1] > y.shape[-1]:
        y = torch.nn.functional.pad(y, (0, x.shape[-1] - y.shape[-1], 0, 0), value=-1)
    return x, y


if __name__ == "__main__":
    from util import plot_task
    import os
    import json

    candidate = Candidate()

    TRAIN_PATH = './ARC/data/training'
    training_tasks = sorted(os.listdir(TRAIN_PATH))
    task = training_tasks[1]
    task_path = os.path.join(TRAIN_PATH, task)
    with open(task_path, 'r') as f:
        task_ = json.load(f)
    task = task_['train']
    score = candidate.evaluate_fitness(task)
    # plot_task(task_)
    print(tuple(score))


















# Functions below are modified from other packages

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

    layers.insert(0, list(set(inputs))) # add input nodes as first layer

    return list(layers)

