from nns.core.layers.layer import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.lastInputShape = None

    def _shape_of(self, inputs):
        if isinstance(inputs, list):
            if len(inputs) == 0:
                return [0]

            return [len(inputs)] + self._shape_of(inputs[0])

        return []

    def forwardPass(self, inputs):
        self.lastInputShape = self._shape_of(inputs)

        def flatten_recursive(x):
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                return sum([flatten_recursive(i) for i in x], [])
            else:
                return x if isinstance(x, list) else [x]
        return flatten_recursive(inputs)

    def backwardPass(self, previousLayerOutputs, expectedOutputsErrorDerivatives=None):
        if expectedOutputsErrorDerivatives is None:
            dL_dout = previousLayerOutputs
            input_shape = self.lastInputShape
        else:
            dL_dout = expectedOutputsErrorDerivatives
            input_shape = self.lastInputShape or self._shape_of(previousLayerOutputs)

        def unflatten(flat, shape):
            if len(shape) == 1:
                return flat[:shape[0]]
            size = shape[0]
            rest = shape[1:]
            chunkSize = int(len(flat) / size) if size != 0 else 0
            return [unflatten(flat[i * chunkSize:(i + 1) * chunkSize], rest) for i in range(size)]
        return unflatten(list(dL_dout), input_shape)
