# def conv2d(matrix: Union[List[List[float]], np.ndarray],
# kernel: Union[List[List[float]], np.ndarray],
# stride: Tuple[int, int] = (1, 1),
# dilation: Tuple[int, int] = (1, 1),
# padding: Tuple[int, int] = (0, 0)) -> np.ndarray:


from typing import List, Tuple
Kernel = List[List[float]]
Stride = Tuple[int, int]
Dilation = Tuple[int, int]
Padding = Tuple[int, int]

def create_zero_matrix(rows: int, cols: int) -> list:
    # Creates matrix filled with zeros
    return [[0 for _ in range(cols)] for _ in range(rows)]

class Convolute2DFunction():
    """
    Parameters:
    -----------
    kernel : 2D list representing the convolution filter.
    stride : A pair of integers telling how many steps (or pixels) the filter moves at each slide over the input.
    dilation : A pair of integers specyfing spacing out the elements of a kernel (filter) to cover a larger area without increasing its size.
    padding : A pair of integers specifying the number of pixels to pad on each side (height, width).
    """
    def __init__(self, kernel, stride, dilation, padding):
        self.name = self.__class__.__name__
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        super().__init__()

    # the input is a matrix
    def call(self, input):
        matrix = input

        kernel_shape_first_dim = len(self.kernel)
        kernel_shape_second_dim = len(self.kernel[0]) if kernel_shape_first_dim > 0 else 0
        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0

        effective_kernel_height = (kernel_shape_first_dim - 1) * self.dilation[0] + 1
        effective_kernel_width = (kernel_shape_second_dim - 1) * self.dilation[1] + 1

        padded_height = n + (2 * self.padding[0])
        padded_width = m + (2 * self.padding[1])

        output_height = (padded_height - effective_kernel_height) // self.stride[0] + 1
        output_width = (padded_width - effective_kernel_width) // self.stride[1] + 1

        output_matrix = create_zero_matrix(output_height, output_width)

        for i in range(output_height):
            for k in range(output_width):
                sum_result = 0

                for a in range(kernel_shape_first_dim):
                    for b in range(kernel_shape_second_dim):
                        inputRowIndex = i * self.stride[0] + a * self.dilation[0] - self.padding[0]
                        inputColumnIndex = k * self.stride[1] + b * self.dilation[1] - self.padding[1]

                        if 0 <= inputRowIndex < n and 0 <= inputColumnIndex < m:
                            sum_result += matrix[inputRowIndex][inputColumnIndex] * self.kernel[a][b]

                output_matrix[i][k] = sum_result

        return output_matrix

    def derivative(self):
        raise TypeError()

    @classmethod
    def fromDict(self, objectDict, _):
        pass
