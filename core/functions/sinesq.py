from .sine import SineFunction

class SineSquaredFunction(SineFunction):
    def __init__(self):
        super().__init__()

        self.name = self.__class__.__name__
    
    def call(self, input):
        return super().call(input) ** 2
    
    def derivative(self, input):
        return 2 * super().call(input) * super().derivative(input)
    
    @classmethod
    def fromDict(self, *_):
        return (self)()