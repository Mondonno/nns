import math

from ....core.functions.sine import SineFunction

class SineLinearFunction(SineFunction):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return input + super().call(input)
    
    def derivative(self, input):
        return 1 + math.cos(input)
    
    @classmethod
    def fromDict(self, *_):
        return (self)()