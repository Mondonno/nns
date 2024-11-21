import math

from .function import Function

class SineFunction(Function):
    def __init__(self, coefficient = 1, bias = 0):
        super().__init__()

        self.name = self.__class__.__name__

        self.coefficient = coefficient
        self.bias = bias

    def call(self, input):
        try:
            result = self.coefficient * math.sin(input) + self.bias
        except Exception as e:
            print("Problem with math domain", input, self.coefficient, self.bias, e)
            raise TypeError()

        return result

    def derivative(self, input):
        return self.coefficient * math.cos(input)
    
    @classmethod
    def fromDict(self, objectDict, _):
        coefficientObject = objectDict["coefficient"]
        coefficient = coefficientObject

        biasObject = objectDict["bias"]
        bias = biasObject
        
        return (self)(coefficient, bias)