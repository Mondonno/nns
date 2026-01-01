class Model():
    def __init__(self) -> None:
        pass

    def fit(self, _):
        raise NotImplementedError("Can not fit a model of type Model")
    
    def forwardPass(self, _):
        raise NotImplementedError("Can not forward pass through a model of type Model")