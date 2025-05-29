import pytest
from .convolute import Convolute2DFunction

def test_basic_convolution():
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    kernel = [
        [1, 0],
        [0, -1]
    ]
    stride = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    conv = Convolute2DFunction(kernel, stride, dilation, padding)
    output = conv.call(matrix)
    assert isinstance(output, list)
    assert len(output) > 0

@pytest.mark.parametrize("stride,padding", [((1, 1), (0, 0)), ((2, 2), (1, 1))])
def test_stride_and_padding(stride, padding):
    matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    kernel = [[1, 0], [0, -1]]
    dilation = (1, 1)
    conv = Convolute2DFunction(kernel, stride, dilation, padding)
    output = conv.call(matrix)
    assert isinstance(output, list)

@pytest.mark.parametrize("dilation", [(1, 1), (2, 2)])
def test_dilation(dilation):
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1, 0], [0, -1]]
    stride = (1, 1)
    padding = (0, 0)
    conv = Convolute2DFunction(kernel, stride, dilation, padding)
    output = conv.call(matrix)
    assert isinstance(output, list)

def test_invalid_derivative():
    kernel = [[1]]
    stride = (1, 1)
    dilation = (1, 1)
    padding = (0, 0)
    conv = Convolute2DFunction(kernel, stride, dilation, padding)
    with pytest.raises(TypeError):
        conv.derivative()
