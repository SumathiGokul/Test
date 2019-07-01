import os
from ctypes import *

LIB_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1])
lib = cdll.LoadLibrary('{}/libmarabou.so'.format(LIB_PATH))

TEST_SET_SIZE = 1000000

def is_candidate(nnet_path: str, test_set_size: int = TEST_SET_SIZE) -> bool:
    '''
    Evaluate the given network with random test set of the given size.
    If found output > 0 return False, otherwise True\
    '''
    result = (c_bool * 1) ()
    p_result = cast(result, POINTER(c_bool))
    c_path = c_char_p(nnet_path.encode('ascii'))
    c_size = c_int(test_set_size)
    ret = lib.is_candidate(c_path, c_size, p_result)

    if ret == -1:
        raise Exception("{} doesn't exist".format(nnet_path))
    if ret == -2:
        raise Exception("Invalid test set size")
    if ret < 0:
        raise Exception()
        
    return result[0]

def get_max(nnet_path: str, test_set_size: int = TEST_SET_SIZE) -> float:
    '''
    Evaluate a random test set and return the max output value
    '''
    result = (c_double * 1)()
    p_result = cast(result, POINTER(c_double))
    c_path = c_char_p(nnet_path.encode('ascii'))
    c_size = c_int(test_set_size)
    ret = lib.get_max(c_path, c_size, p_result)

    if ret == -1:
        raise Exception("{} doesn't exist".format(nnet_path))
    if ret == -2:
        raise Exception("Invalid test set size")
    if ret < 0:
        raise Exception()
        
    return result[0]

def diff_nnet(nnet1: str, nnet2: str, test_set_size: int = TEST_SET_SIZE) -> float:
    '''
    Evaluate the given networks with a random test set of the give size.
    Returns the number of times the network evaluated diffrently devided by
    the test set size.
    '''
    result = (c_double * 1)()
    p_result = cast(result, POINTER(c_double))
    c_path1 = c_char_p(nnet1.encode('ascii'))
    c_path2 = c_char_p(nnet2.encode('ascii'))
    c_size = c_int(test_set_size)
    ret = lib.diff_nnet(c_path1, c_path2, c_size, p_result)

    if ret == -1:
        raise Exception("{} doesn't exist".format(nnet1))
    if ret == -2:
        raise Exception("{} doesn't exist".format(nnet2))
    if ret == -3:
        raise Exception("Invalid test set size")
    if ret < 0:
        raise Exception()


    return result[0]

def eval_node(nnet: str, zero: float = 0.000001):
    '''
    Runs marabou verification process on the given network.
    Input bounds are set to the network's training mins and maxes.
    Output bounds are set to the given zero and DBL_MAX.
    Returns True if the query is unsat, otherwise False.
    '''
    result = (c_bool * 1)()
    p_result = cast(result, POINTER(c_bool))
    c_nnet = c_char_p(nnet.encode('ascii'))
    c_zero = c_double(zero)
    ret = lib.eval_node(c_nnet, c_zero, p_result)

    if ret == -1:
        raise Exception("{} doesn't exist".format(nnet))
    if ret == -2:
        raise Exception("Network must have only 1 output node")
    if ret == -3:
        raise Exception("ReluplexError")
    if ret < 0:
        raise Exception()

    return result[0]
 

def eval_node_bounds(nnet: str, bounds: str) -> bool:
    '''
    Runs marabou verification process on the given network with the
    given boundaries.
    Returns True if the query is unsat, otherwise False.
    '''
    result = (c_bool * 1)()
    p_result = cast(result, POINTER(c_bool))
    c_nnet = c_char_p(nnet.encode('ascii'))
    c_bounds = c_char_p(bounds.encode('ascii'))
    ret = lib.eval_node_bounds(c_nnet, c_bounds, p_result)

    if ret == -1:
        raise Exception("{} doesn't exist".format(nnet))
    if ret == -2:
        raise Exception("{} doesn't exist".format(bounds))
    if ret == -3:
        raise Exception("Failed to setup boundaries")
    if ret == -4:
        raise Exception("ReluplexError")
    if ret < 0:
        raise Exception()

    return result[0]
