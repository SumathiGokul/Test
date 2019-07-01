from typing import Tuple, List, Dict
from pathlib import Path
import numpy as np
import logging
import copy
import os
import sys
from .AcasNnet import AcasNnet
import time


DELIMITER = ','
WEIGHTS = 'weights'
BIAS = 'bias'
COMMENT_PREFIX = '//'

MAX_FLOAT: float = sys.float_info.max
ZERO: float = 0.000001


formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def load_nnet(nnet_path: str) -> AcasNnet:
    """ Based on load_network(const char* filename) in AcasNnet.cpp """
    logger.debug("loading %s", nnet_path)

    nnet = AcasNnet()

    nnet.name = os.path.basename(nnet_path).split('.')[0]

    with open(nnet_path, 'r') as nnet_file:
        # read header
        line = nnet_file.readline()
        while line.strip()[0:2] == COMMENT_PREFIX:
            nnet.header.append(line)
            line = nnet_file.readline()

        nnet.num_layers, nnet.input_size, nnet.output_size, nnet.max_layer_size = [int(x) for x in line.split(DELIMITER)[:4]]
        logger.debug("number of layers: {}".format(nnet.num_layers))
        logger.debug("input size: {}".format(nnet.input_size))
        logger.debug("output size: {}".format(nnet.output_size))
        logger.debug("max layer size: {}".format(nnet.max_layer_size))

        nnet.layer_sizes = list([int(x) for x in nnet_file.readline().split(DELIMITER)[:nnet.num_layers + 1]])
        logger.debug("layer sizes: {}".format(nnet.layer_sizes))

        nnet.symmetric = int(nnet_file.readline()[:1])
        logger.debug("symmetric: {}".format(nnet.symmetric))

        nnet.mins = tuple([float(x) for x in nnet_file.readline().split(DELIMITER)[:nnet.input_size]])
        logger.debug("mins: {}".format(nnet.mins))

        nnet.maxes = tuple([float(x) for x in nnet_file.readline().split(DELIMITER)[:nnet.input_size]])
        logger.debug("maxes: {}".format(nnet.maxes))

        nnet.means = tuple([float(x) for x in nnet_file.readline().split(DELIMITER)[:nnet.input_size + 1]])
        logger.debug("means: {}".format(nnet.means))

        nnet.ranges = tuple([float(x) for x in nnet_file.readline().split(DELIMITER)[:nnet.input_size + 1]])
        logger.debug("ranges: {}".format(nnet.ranges))

        for i in range(nnet.num_layers):
            nnet.matrix.append(dict())
            cols = nnet.layer_sizes[i]
            rows = nnet.layer_sizes[i + 1]
            nnet.matrix[i][WEIGHTS] = np.zeros([rows, cols], dtype=np.float64)
            nnet.matrix[i][BIAS] = np.zeros([rows, 1], dtype=np.float64)
            logger.debug("layer {}, weights shape {}, bias shape {}".format(i, nnet.matrix[i][WEIGHTS].shape,
                                                                            nnet.matrix[i][BIAS].shape))

        layer = 0
        param = WEIGHTS
        i = 0
        j = 0
        for line in nnet_file.readlines():
            if i >= nnet.layer_sizes[layer + 1]:
                if param == WEIGHTS:
                    param = BIAS
                else:
                    param = WEIGHTS
                    layer += 1
                i = 0
                j = 0
            for val in line.split(DELIMITER)[:-1]:
                nnet.matrix[layer][param][i][j] = float(val)
                j += 1

            j = 0
            i += 1

    nnet.inputs = np.zeros(nnet.max_layer_size, dtype=np.float64)
    nnet.temp = np.zeros(nnet.max_layer_size, dtype=np.float64)
    return nnet

def evaluate_network(nnet: AcasNnet, inputs: List[float], normalizeInput, normalizeOutput) -> List[float]:
    """ Based on evaluate_network in AcasNnet.cpp """
    num_layers = nnet.num_layers 
    output_size = nnet.output_size
    input_size = nnet.input_size
    symmetric = nnet.symmetric

    matrix = nnet.matrix
    output = np.zeros(output_size, dtype=np.float64)

    if (normalizeInput):
        for i in range(input_size):
            if (inputs[i] > nnet.maxes[i]):
                nnet.inputs[i] = (nnet.maxes[i] - nnet.means[i]) / (nnet.ranges[i])
            elif (inputs[i] < nnet.mins[i]):
                nnet.inputs[i] = (nnet.mins[i] - nnet.means[i]) / (nnet.ranges[i])
            else:
                nnet.inputs[i] = (inputs[i] - nnet.means[i]) / (nnet.ranges[i])
        if (symmetric == 1 and nnet.inputs[2] < 0):
            nnet.inputs[2] = -nnet.inputs[2] #Make psi positive
            nnet.inputs[1] = -nnet.inputs[1] #Flip across x-axis
        else:
            symmetric = 0
    else:
        for i, val in enumerate(inputs):
            nnet.inputs[i] = val

    for layer in range(num_layers):
        weights = matrix[layer][WEIGHTS]
        biases = matrix[layer][BIAS]
        for i in range(nnet.layer_sizes[layer + 1]):
            temp_val = 0.0

            # weight summation of inputs
            for j in range(nnet.layer_sizes[layer]):
                temp_val += nnet.inputs[j] * weights[i][j]

            # add bias to weighted sum
            temp_val += biases[i][0]

            # perform ReLU
            if temp_val < 0.0 and layer < num_layers - 1:
                temp_val = 0.0
            nnet.temp[i] = temp_val

        for i in range(nnet.layer_sizes[layer + 1]):
            nnet.inputs[i] = nnet.temp[i]

    for i in range(output_size):
        if(normalizeOutput):
            output[i] = nnet.inputs[i] * nnet.ranges[nnet.input_size] + nnet.means[nnet.input_size]
        else:
            output[i] = nnet.inputs[i]

    if (symmetric == 1):
        temp_value = output[1]
        output[1] = output[2]
        output[2] = temp_value
        temp_value = output[3]
        output[3] = output[4]
        output[4] = temp_value

    return output


def save_network(nnet: AcasNnet, path: str):
    """ save the given network with AcasNnet file format """
    with open(path, mode='w') as f:
        for line in nnet.header:
            f.write(line)
        f.write('{},{},{},{},\n'.format(nnet.num_layers, nnet.input_size,
                                        nnet.output_size, nnet.max_layer_size))
        for layer_size in nnet.layer_sizes:
            f.write('{},'.format(layer_size))
        f.write('\n0,\n')  # symmetric
        for val in nnet.mins:
            f.write('{},'.format(val))
        f.write('\n')
        for val in nnet.maxes:
            f.write('{},'.format(val))
        f.write('\n')
        for val in nnet.means:
            f.write('{},'.format(val))
        f.write('\n')
        for val in nnet.ranges:
            f.write('{},'.format(val))
        f.write('\n')

        for layer in nnet.matrix:
            for weights in layer['weights']:
                for weight in weights:
                    f.write('{},'.format(weight))
                f.write('\n')
            for bias in layer['bias']:
                f.write('{},\n'.format(bias[0]))


def deconstruct_layer(nnet: AcasNnet, layer) -> List[AcasNnet]:
    nets = list()
    for node in range(nnet.layer_sizes[layer + 1]):
        sub_net = copy.copy(nnet)
        sub_net.layer_sizes = nnet.layer_sizes[:layer + 1] + [1]
        sub_net.num_layers = len(sub_net.layer_sizes) - 1
        sub_net.output_size = 1
        sub_net.max_layer_size = max(sub_net.layer_sizes)

        sub_net.matrix = list()
        for i in range(layer):
            sub_net.matrix.append(dict())
            sub_net.matrix[i]['weights'] = nnet.matrix[i]['weights']
            sub_net.matrix[i]['bias'] = nnet.matrix[i]['bias']

        sub_net.matrix.append(dict())
        sub_net.matrix[layer]['weights'] = list()
        sub_net.matrix[layer]['bias'] = list()
        sub_net.matrix[layer]['weights'].append(nnet.matrix[layer]['weights'][node])
        sub_net.matrix[layer]['bias'].append(nnet.matrix[layer]['bias'][node])
        nets.append(sub_net)

    return nets


def remove_node(nnet: AcasNnet, layer: int, node: int) -> AcasNnet:
    simplified_nnet = copy.deepcopy(nnet)
    simplified_nnet.matrix[layer]['weights'] = np.delete(simplified_nnet.matrix[layer]['weights'], node, 0)
    simplified_nnet.matrix[layer]['bias'] = np.delete(simplified_nnet.matrix[layer]['bias'], node, 0)
    if layer < nnet.num_layers - 1:
        simplified_nnet.matrix[layer + 1]['weights'] = np.delete(simplified_nnet.matrix[layer + 1]['weights'], node, 1)
    simplified_nnet.layer_sizes[layer + 1] -= 1
    simplified_nnet.max_layer_size = max(simplified_nnet.layer_sizes)
    return simplified_nnet


def create_bounds(nnet: AcasNnet, path) -> None:
    bounds_str = 'OUT 0 LOW {:.10f}\n'.format(ZERO)
    bounds_str += 'OUT 0 UP {}\n'.format(MAX_FLOAT)
    for i in range(nnet.input_size):
        bounds_str += 'IN {} LOW {}\n'.format(i, nnet.mins[i])
        bounds_str += 'IN {} UP {}\n'.format(i, nnet.maxes[i])
    with open(path, 'w') as f:
        f.write(bounds_str)
