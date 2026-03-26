import math

from .function import Function


class SoftmaxFunction(Function):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__

    def call(self, input):
        return input

    def callVector(self, inputs):
        maxInput = max(inputs)
        exponentials = [math.exp(singleInput - maxInput) for singleInput in inputs]
        exponentialsSum = sum(exponentials)
        return [singleExponential / exponentialsSum for singleExponential in exponentials]

    def jacobian(self, inputs):
        probabilities = self.callVector(inputs)
        jacobianMatrix = []

        for rowIndex in range(len(probabilities)):
            jacobianRow = []

            for columnIndex in range(len(probabilities)):
                kroneckerDelta = 1 if rowIndex == columnIndex else 0
                jacobianRow.append(
                    probabilities[rowIndex] * (kroneckerDelta - probabilities[columnIndex])
                )

            jacobianMatrix.append(jacobianRow)

        return jacobianMatrix

    def derivative(self, input):
        _ = input
        raise TypeError("Softmax derivative is vector-valued. Use jacobian() instead.")

    @classmethod
    def fromDict(self, *_):
        return (self)()
