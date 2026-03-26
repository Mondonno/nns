import math

from .function import Function


class CrossEntropyFunction(Function):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.name = self.__class__.__name__
        self.epsilon = epsilon

    def _clip(self, value):
        return min(max(value, self.epsilon), 1 - self.epsilon)

    def call(self, input):
        prediction, target = input

        if target == 0:
            return 0

        return -(target * math.log(self._clip(prediction)))

    def derivative(self, input):
        prediction, target = input

        if target == 0:
            return 0

        return -(target / self._clip(prediction))

    @classmethod
    def fromDict(self, objectDict, _):
        epsilon = objectDict.get("epsilon", 1e-12)
        return (self)(epsilon=epsilon)
