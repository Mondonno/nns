
from .....core.functions.function import Function

class CustomLearningRateFunction(Function):
    def __init__(self):
        super().__init__()
    
    def call(self, epochIndex):
        # 0.00008
        # 0.0009 * 1/(1 + (1.009 ** epochIndex))
        # 0.00009 / (1 + (epochIndex / 100))
        # / (1 + (epochIndex / 18))
        return 0.0003
    
    @classmethod
    def fromDict(self, *_):
        return (self)()
