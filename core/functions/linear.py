from .function import Function

class LinearFunction(Function):
    def __init__(self, coefficient = 1, bias = 0):
        super().__init__()

        self.name = self.__class__.__name__
        
        self.coefficient = coefficient
        self.bias = bias
    
    def call(self, input):
        return self.coefficient * input + self.bias
    
    def derivative(self, _):
        return self.coefficient
    
    @classmethod
    def fromDict(self, objectDict, _):
        coefficientObject = objectDict["coefficient"]
        coefficient = coefficientObject

        biasObject = objectDict["bias"]
        bias = biasObject
        
        return (self)(coefficient, bias)

