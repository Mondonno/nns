from nns.core.layers.layer import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forwardPass(self, inputs):
        # Flattens input of shape [channels][height][width] or [height][width] to 1D list
        def flatten_recursive(x):
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                return sum([flatten_recursive(i) for i in x], [])
            else:
                return x if isinstance(x, list) else [x]
        return flatten_recursive(inputs)

    def backwardPass(self, dL_dout, input_shape):
        # Reshape 1D gradient back to input shape
        def unflatten(flat, shape):
            if len(shape) == 1:
                return flat[:shape[0]]
            size = shape[0]
            rest = shape[1:]
            return [unflatten(flat[i*int(len(flat)/size):(i+1)*int(len(flat)/size)], rest) for i in range(size)]
        return unflatten(dL_dout, input_shape)