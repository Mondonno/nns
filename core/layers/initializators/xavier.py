import math
import random

from random import Random

from ...functions.linear import Function

class XavierInitializatorFunction(Function):
    def __init__(self, seed = None) -> None:
        self.name = self.__class__.__name__

        if seed is None:
            self.seed = random.Random(seed).random()
        else:
            self.seed = seed

        self.random = Random(seed)

    def call(self, layer, neuronIndex, inputIndex):
        print("NEURON INDEX/INPUTINDEX")
        print(neuronIndex, inputIndex, layer.inputsCount)

        # inputsCount is not -1 because it does not include bias
        if inputIndex == (layer.inputsCount): # it is a bias, xavier sets biases to zero
            return 0

        xavierExpression = math.sqrt(6) / math.sqrt(layer.inputsCount + layer.neuronsCount)
        return xavierExpression * self.random.random()
    
    @classmethod
    def fromDict(self, objectDict, additionalDict):
        referenceDict = additionalDict
        
        _ = referenceDict

        seed = objectDict["seed"]

        return XavierInitializatorFunction(seed=seed)