from .layer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.lastInputs = None

    def _mapRecursive(self, values, mapper):
        if isinstance(values, list):
            return [self._mapRecursive(singleValue, mapper) for singleValue in values]

        return mapper(values)

    def forwardPass(self, inputs):
        self.lastInputs = inputs
        return self._mapRecursive(inputs, self.activation.call)

    def backwardPass(self, previousLayerOutputs, expectedOutputsErrorDerivatives=None):
        _ = previousLayerOutputs

        if expectedOutputsErrorDerivatives is None:
            dL_dout = previousLayerOutputs
        else:
            dL_dout = expectedOutputsErrorDerivatives

        def mapper(singleInput, singleGradient):
            if isinstance(singleInput, list):
                return [
                    mapper(singleInput[index], singleGradient[index])
                    for index in range(len(singleInput))
                ]

            return self.activation.derivative(singleInput) * singleGradient

        return mapper(self.lastInputs, dL_dout)
