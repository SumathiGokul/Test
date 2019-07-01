from typing import List, Tuple, Dict, Any

class AcasNnet:
    """ Based on AcasNnet class in AcasNnet.h """
    def __init__(self):
        self.name: str
        self.header: List[str] = list()
        self.symmetric: int
        self.num_layers: int
        self.input_size: int
        self.output_size: int
        self.max_layer_size: int
        self.layer_sizes: List[int]

        self.mins: Tuple[float]
        self.maxes: Tuple[float]
        self.means: Tuple[float]
        self.ranges: Tuple[float]
        self.matrix: List[Dict[str, Any]] = list()

        self.inputs = None # An array
        self.temp = None

