from ....core.functions.linear import LinearFunction

class CustomLinearFunction(LinearFunction):
    def __init__(self):
        super().__init__()

        self.coefficient = 4
        self.bias = 3
