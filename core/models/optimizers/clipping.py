import math

from .function import OptimizerFunction

class ClippingOptimizerFunction(OptimizerFunction):
    def __init__(self, threshold = 5e3):
        super().__init__()

        self.threshold = threshold

    def call(self, gradientValue, gradientVector):
        gradientNorm = 0 # L1
        for singleGradientNorm in gradientVector:
            gradientNorm += math.fabs(singleGradientNorm)
        
        modifiedGradientValue = gradientValue

        if gradientNorm > self.threshold:
            print("Gradient clipped")
            clipFactor = self.threshold / gradientNorm
            modifiedGradientValue = modifiedGradientValue * clipFactor

        return modifiedGradientValue
