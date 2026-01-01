from ...functions import Function

class OptimizerFunction(Function):
    def __init__(self):
        super().__init__()

    def call(self, gradientValue, gradientVector):
        raise NotImplementedError("Optimizer function must have call method defined")
    
    def derivative(self):
        raise NotImplementedError("Optimizer function shouldn't have derivative defined")