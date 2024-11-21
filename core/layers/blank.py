import math

from .dense import Dense

from ..functions.linear import LinearFunction

class Blank(Dense):
    def __init__(self, inputsCount, neuronsCount):
        super().__init__(inputsCount, neuronsCount, activation = LinearFunction(), seed = math.nan)
        self.name = self.__class__.__name__

        self.weightsCount = 0