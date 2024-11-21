from ....core.functions.sine import SineFunction

class CustomSineFunction(SineFunction):
    def __init__(self):
        super().__init__()

        self.coefficient = 7
        self.bias = 6

class CustomSineSimplifiedFunction(SineFunction):
    def __init__(self):
        super().__init__()

        self.coefficient = 7
