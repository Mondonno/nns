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
        # we can skip checkin the correctness of params at the moment
        matrix = input

        kernel_shape_first_dim = len(self.kernel)
        kernel_shape_second_dim = len(self.kernel[0]) if kernel_shape_first_dim > 0 else 0
        kernel_shape = (kernel_shape_first_dim, kernel_shape_second_dim)


        n = len(matrix)
        m = len(matrix[0]) if n > 0 else 0

        effective_kernel_height = (kernel_shape[0] - 1) * self.dilation[0] - 1
        effective_kernel_width = (kernel_shape[1] - 1) * self.dilation[1] - 1

        padded_height = n + (2 * self.padding[0])
        padded_width = m + (2 * self.padding[1])

        output_height = (padded_height - effective_kernel_height) // self.stride[0] + 1
        output_width = (padded_width - effective_kernel_width) // self.stride[1] + 1

        output_matrix = create_zero_matrix(output_height, output_width)

        kernel_offset = (kernel_shape[0] // 2, kernel_shape[1] // 2)

        center_x_starting_point = kernel_offset[0] * self.dilation[0]
        center_y_starting_point = kernel_offset[1] * self.dilation[1]

        for i in range(output_height):
            center_x = center_x_starting_point + (i * self.stride[0])

            # Calculate the horizontal indices (columns) of the input matrix that the kernel will cover during convolution.
            # The kernel is centered at `center_x`, and its coverage is expanded by the dilation factor along the x-axis.
            # The `kernel_offset[0]` represents the half-width of the kernel, and `self.dilation[0]` controls how much
            # the kernel elements are spaced apart.
            # For each position in the kernel, `j` ranges from -kernel_offset[0] to kernel_offset[0], which gives us the
            # relative positions of the kernel's elements.
            # The dilation factor is applied by multiplying `j` with `self.dilation[0]`, effectively increasing the spacing
            # between adjacent elements in the kernel.
            indices_x = [(center_x + j * self.dilation[0]) for j in range(-kernel_offset[0], kernel_offset[0] + 1)]
            for k in range(output_width):
                center_y = center_y_starting_point + (k * self.stride[1]) # it repeats we can abstract this into multiple dimensions
                indices_y = [center_y + j * self.dilation[1] for j in range(-kernel_offset[1], kernel_offset[1] + 1)]

                # Extract submatrix using indices_x and indices_y for pure Python lists
                submatrix = [
                    [matrix[x][y] if 0 <= x < n and 0 <= y < m else 0 for y in indices_y]
                    for x in indices_x
                ]

                sum_result = 0
                for a in range(len(self.kernel)):
                    for b in range(len(self.kernel[0])):
                        sum_result += submatrix[a][b] * self.kernel[a][b]

                output_matrix[i][k] = sum_result

        return output_matrix # it should do it :)

    def derivative(self):
        raise TypeError()

    @classmethod
    def fromDict(self, objectDict, _):
        pass