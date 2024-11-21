from .function import Function

class MSEFunction(Function):
    def __init__(self):
        self.name = self.__class__.__name__
        super().__init__()
    
    def call(self, input):
        try:
            return ((input[0] - input[1]) ** 2)
        except Exception as e:      
            print("Problem with math domain", input, e)
            raise

    def derivative(self, input):
        return 2 * (input[0] - input[1])
    
    @classmethod
    def fromDict(self, *_):
        return (self)()