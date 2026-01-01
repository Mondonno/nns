from nns.core.functions.convolute import Convolute2DFunction
from nns.core.layers.layer import Layer
import random

class Convolution2D(Layer):
    def __init__(self, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0), seed=None):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        if seed is None:
            self.seed = random.Random(seed).random()
        else:
            self.seed = seed
            
        randomInstance = random.Random(self.seed)

        # Randomly initialize kernels for each output channel using randomInstance
        self.kernels = [
            [[randomInstance.uniform(-0.1, 0.1) for _ in range(self.kernel_size[1])] for _ in range(self.kernel_size[0])]
            for _ in range(self.out_channels)
        ]
        self.neuronsCount, self.inputsCount, self.weights = self._prepareWeights(self.kernels)
        self.weightsCount = self.inputsCount + 1

    def _prepareWeights(self, weights):
        """Prepare weights as [neuronsCount][inputsCount + 1] (last is bias) and return (neuronsCount, inputsCount, weightsList)"""
        if isinstance(weights[0][0], list):
            neuronsCount = len(weights)
            kernelHeight = len(weights[0])
            kernelWidth = len(weights[0][0])
            inputsCount = kernelHeight * kernelWidth
            weightsList = []
            for channel in weights:
                flat_kernel = [v for row in channel for v in row]
                weightsList.append(flat_kernel + [0.0])
        else:
            neuronsCount = 1
            kernelHeight = len(weights)
            kernelWidth = len(weights[0])
            inputsCount = kernelHeight * kernelWidth
            flat_kernel = [v for row in weights for v in row]
            weightsList = [flat_kernel + [0.0]]
        return neuronsCount, inputsCount, weightsList

    def _countWeights(self):
        # Count total weights in the kernel(s)
        if isinstance(self.weights[0][0], list):
            # Multi-channel: sum over all channels
            return sum(len(channel) * len(channel[0]) for channel in self.weights)
        else:
            return len(self.weights) * len(self.weights[0])

    def forwardPass(self, inputs, debug=False):
        # Support for batch and channel dimensions
        # inputs: [batch][channel][height][width] or [channel][height][width]
        # For compatibility, always return (filledInputs, outputs)
        if isinstance(inputs[0][0][0], list):  # batch mode
            outputs = [self._forwardSingle(sample) for sample in inputs]
            filledInputs = inputs
        else:
            outputs = self._forwardSingle(inputs)
            filledInputs = inputs
        return filledInputs, outputs

    def _forwardSingle(self, inputs):
        def reconstruct_kernel(kernel_flat, input_shape):
            kernel_side = input_shape[0]
            return [kernel_flat[j * kernel_side:(j + 1) * kernel_side] for j in range(kernel_side)]

        if self.neuronsCount > 1:
            outputs = []
            for i in range(self.neuronsCount):
                kernel_flat = self.weights[i][:-1]
                bias = self.weights[i][-1]
                kernel = reconstruct_kernel(kernel_flat, (len(inputs[0]), len(inputs[0][0]) if len(inputs[0]) > 0 else 1))
                conv = Convolute2DFunction(kernel, self.stride, self.dilation, self.padding)
                out = conv.call(inputs[i])
                if isinstance(out[0], list):
                    out = [[v + bias for v in row] for row in out]
                else:
                    out = [v + bias for v in out]
                outputs.append(out)
            return outputs
        else:
            kernel_flat = self.weights[0][:-1]
            bias = self.weights[0][-1]
            kernel = reconstruct_kernel(kernel_flat, (len(inputs), len(inputs[0]) if len(inputs) > 0 else 1))
            conv = Convolute2DFunction(kernel, self.stride, self.dilation, self.padding)
            out = conv.call(inputs)
            if isinstance(out[0], list):
                out = [[v + bias for v in row] for row in out]
            else:
                out = [v + bias for v in out]
            return out

    def backwardPass(self, previousLayerOutputs, expectedOutputsErrorDerivatives, learningRate=0.01, debug=False):
        """
        previousLayerOutputs: original input to this layer (needed for kernel gradient)
        expectedOutputsErrorDerivatives: gradient of loss w.r.t. output of this layer (same shape as output)
        learningRate: step size for updating kernel
        Returns: (layerWeightsDerivativesVector, nextLayerErrorDerivatives)
        """
        def reconstruct_kernel(kernel_flat, input_shape):
            kernel_side = input_shape[0]
            return [kernel_flat[j * kernel_side:(j + 1) * kernel_side] for j in range(kernel_side)]

        dL_dout = expectedOutputsErrorDerivatives
        layerWeightsDerivativesVector = []

        if self.neuronsCount > 1:
            gradientsWrtInputs = []
            for i in range(self.neuronsCount):
                kernel_flat = self.weights[i][:-1]
                bias = self.weights[i][-1]
                input_shape = (len(previousLayerOutputs[i]), len(previousLayerOutputs[i][0]) if len(previousLayerOutputs[i]) > 0 else 1)
                kernel = reconstruct_kernel(kernel_flat, input_shape)

                flippedKernel = self._flipWeights(kernel)
                convolutionFunctionForInput = Convolute2DFunction(flippedKernel, (1, 1), (1, 1), (0, 0))
                gradientWrtInput = convolutionFunctionForInput.call(dL_dout[i])
                gradientsWrtInputs.append(gradientWrtInput)

                convolutionFunctionForWeights = Convolute2DFunction(dL_dout[i], (1, 1), (1, 1), (0, 0))
                gradientWrtKernel = convolutionFunctionForWeights.call(previousLayerOutputs[i])
                grad_flat = [v for row in gradientWrtKernel for v in row]
                if isinstance(dL_dout[i][0], list):
                    grad_bias = sum(sum(row) for row in dL_dout[i])
                else:
                    grad_bias = sum(dL_dout[i])
                layerWeightsDerivativesVector.extend(grad_flat + [grad_bias])
            nextLayerErrorDerivatives = gradientsWrtInputs
        else:
            kernel_flat = self.weights[0][:-1]
            bias = self.weights[0][-1]
            input_shape = (len(previousLayerOutputs), len(previousLayerOutputs[0]) if len(previousLayerOutputs) > 0 else 1)
            kernel = reconstruct_kernel(kernel_flat, input_shape)

            flippedKernel = self._flipWeights(kernel)
            convolutionFunctionForInput = Convolute2DFunction(flippedKernel, (1, 1), (1, 1), (0, 0))
            gradientWrtInput = convolutionFunctionForInput.call(dL_dout)

            convolutionFunctionForWeights = Convolute2DFunction(dL_dout, (1, 1), (1, 1), (0, 0))
            gradientWrtKernel = convolutionFunctionForWeights.call(previousLayerOutputs)
            grad_flat = [v for row in gradientWrtKernel for v in row]
            if isinstance(dL_dout[0], list):
                grad_bias = sum(sum(row) for row in dL_dout)
            else:
                grad_bias = sum(dL_dout)
            layerWeightsDerivativesVector.extend(grad_flat + [grad_bias])
            nextLayerErrorDerivatives = gradientWrtInput

        return layerWeightsDerivativesVector, nextLayerErrorDerivatives

    def _flipWeights(self, weights):
        # Flip the weights by 180 degrees (vertical and horizontal)
        return [weightsRow[::-1] for weightsRow in weights[::-1]]