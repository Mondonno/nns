from ...functions import Function

class InitializatorFunction(Function):
    def __init__(self):
        super().__init__()

    def call(self, layer, neuronIndex, inputIndex):
        raise NotImplementedError("Initializator function must have call method defined")
    
    def derivative(self):
        raise NotImplementedError("Initializator function shouldn't be implemented")