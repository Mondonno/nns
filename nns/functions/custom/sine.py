from ....core.functions.sine import SineFunction

class CustomSineFunction(SineFunction):
    def __init__(self):
        super().__init__()

        self.coefficient = 1
        self.bias = 10

class CustomSineSimplifiedFunction(SineFunction):
    def __init__(self):
        super().__init__()

        self.coefficient = 7
