from .function import OptimizerFunction

class MomentumOptimizerFunction(OptimizerFunction):
    def __init__(self):
        super().__init__()

        # m(t+1) = B * mt + (1 - B) * grad

        self.beta = 0.9
        self.momentum = 0

    def call(self, gradientValue, _):
        self.momentum = self.beta * self.momentum + (1 - self.beta) * gradientValue

        return self.momentum