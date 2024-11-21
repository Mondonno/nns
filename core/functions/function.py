class Function():
    def __init__(self):
        self.name = self.__class__.__name__
        pass
    
    def call(self, input):
        raise Exception(f"Couldn't call Function with {input} without function")
    
    def derivative(self, input):
        raise Exception(f"Couldn't derivate Function with {input} without derivation")
