from .function import Function

class RectifiedLinearFunction(Function):
    def __init__(self, coefficient = 1, bias = 0):
        super().__init__()

        self.name = self.__class__.__name__

        self.coefficient = coefficient
        self.bias = bias
    
    def call(self, input):
        if input > 0:
            return self.coefficient * input + self.bias
        else:
            return 0
    
    def derivative(self, input):
        if input > 0:
            return self.coefficient
        else:
            return 0
    
    @classmethod
    def fromDict(self, objectDict, _):
        coefficientObject = objectDict["coefficient"]
        coefficient = coefficientObject

        biasObject = objectDict["bias"]
        bias = biasObject
        
        return (self)(coefficient, bias)