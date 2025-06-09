
from ...functions import Function

class MinMaxTransformFunction(Function):
    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__()

    def __call__(self, value: float) -> float:
        return (value - self.min_value) / (self.max_value - self.min_value)