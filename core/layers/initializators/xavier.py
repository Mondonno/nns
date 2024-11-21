import math

from random import Random

from ...functions.linear import Function

class XavierInitializatorFunction(Function):
    def __init__(self, seed = None) -> None:
        self.name = self.__class__.__name__

        self.seed = seed
        self.random = Random(seed)

    def call(self, layer, neuronIndex, inputIndex):
        if inputIndex == (layer.inputsCount - 1): # it is a bias, xavier sets biases to zero
            return 0

        xavierExpression = math.sqrt(6) / math.sqrt(layer.inputsCount + layer.neuronsCount)
        return xavierExpression * self.random.random()
    
    @classmethod
    def fromDict(self, objectDict, additionalDict):
        referenceDict = additionalDict
        
        _ = referenceDict

        seed = objectDict["seed"]

        return XavierInitializatorFunction(seed=seed)