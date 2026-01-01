import pytest
from nns.core.layers.convolution2d import Convolution2D

class DummyKernel:
    def __init__(self, value=1):
        self.value = value
    def __getitem__(self, idx):
        return [self.value, self.value]
    def __len__(self):
        return 2

def test_convolution2d_single_channel():
    kernel = [[1, 0], [0, -1]]
    layer = Convolution2D(kernel=kernel, stride=(1, 1), dilation=(1, 1), padding=(0, 0))
    inputs = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    output = layer.forwardPass(inputs)
    assert isinstance(output, list)
    assert len(output) == 2
    assert len(output[0]) == 2
    # Check a known value
    assert output[0][0] == 1*1 + 2*0 + 4*0 + 5*(-1)

def test_convolution2d_multi_channel():
    kernel = [[1, 0], [0, -1]]
    layer = Convolution2D(kernel=kernel, stride=(1, 1), dilation=(1, 1), padding=(0, 0))
    inputs = [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]
    ]
    output = layer.forwardPass(inputs)
    assert isinstance(output, list)
    assert len(output) == 2  # two channels
    assert all(isinstance(o, list) for o in output)
    assert all(len(o) == 2 for o in output)

def test_convolution2d_batch():
    kernel = [[1, 0], [0, -1]]
    layer = Convolution2D(kernel=kernel, stride=(1, 1), dilation=(1, 1), padding=(0, 0))
    batch_inputs = [
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ],
        [
            [
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]
            ]
        ]
    ]
    output = layer.forwardPass(batch_inputs)
    assert isinstance(output, list)
    assert len(output) == 2  # batch size
    assert all(isinstance(o, list) for o in output)
    assert all(len(o) == 1 for o in output)  # single channel per sample

def test_convolution2d_backward_single_channel():
    kernel = [[1, 0], [0, -1]]
    layer = Convolution2D(kernel=[[1, 0], [0, -1]])
    inputs = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    dL_dout = [
        [1, 1],
        [1, 1]
    ]
    import copy
    original_kernel = copy.deepcopy(layer.kernel)
    grad_input = layer.backwardPass(dL_dout, inputs, learning_rate=0.1)
    assert isinstance(grad_input, list)
    assert len(grad_input) == 3
    assert len(grad_input[0]) == 3
    assert layer.kernel != original_kernel

def test_convolution2d_backward_multi_channel():
    kernel = [[1, 0], [0, -1]]
    layer = Convolution2D(kernel=[[1, 0], [0, -1]])
    inputs = [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ],
        [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]
    ]
    dL_dout = [
        [
            [1, 1],
            [1, 1]
        ],
        [
            [2, 2],
            [2, 2]
        ]
    ]
    import copy
    original_kernel = copy.deepcopy(layer.kernel)
    grad_inputs = layer.backwardPass(dL_dout, inputs, learning_rate=0.1)
    assert isinstance(grad_inputs, list)
    assert len(grad_inputs) == 2
    assert all(isinstance(gi, list) for gi in grad_inputs)
    assert layer.kernel != original_kernel


