import random

from nns.core.functions.convolute import Convolute2DFunction
from nns.core.layers.layer import Layer


class Convolution2D(Layer):
    def __init__(self, out_channels=None, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 0), seed=None, kernel=None):
        super().__init__()

        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = None

        if seed is None:
            self.seed = random.Random(seed).random()
        else:
            self.seed = seed

        if self.kernel is not None:
            if isinstance(self.kernel[0][0], list):
                self.fixedKernels = self.kernel
            else:
                self.fixedKernels = [self.kernel]

            self.out_channels = len(self.fixedKernels)
            self.kernel_size = (len(self.fixedKernels[0]), len(self.fixedKernels[0][0]))
            self.neuronsCount = self.out_channels
            self.inputsCount = self.kernel_size[0] * self.kernel_size[1]
            self.weights = [[value for row in singleKernel for value in row] + [0.0] for singleKernel in self.fixedKernels]
        else:
            self.fixedKernels = None
            self.neuronsCount = self.out_channels
            self.inputsCount = 0
            self.weights = []

        self.weightsCount = self.inputsCount + 1 if self.inputsCount > 0 else 0

    def _is_batch_input(self, inputs):
        return (
            isinstance(inputs, list) and
            len(inputs) > 0 and
            isinstance(inputs[0], list) and
            len(inputs[0]) > 0 and
            isinstance(inputs[0][0], list) and
            len(inputs[0][0]) > 0 and
            isinstance(inputs[0][0][0], list)
        )

    def _has_channel_dimension(self, inputs):
        return (
            isinstance(inputs, list) and
            len(inputs) > 0 and
            isinstance(inputs[0], list) and
            len(inputs[0]) > 0 and
            isinstance(inputs[0][0], list)
        )

    def _normalize_channels(self, inputs):
        if self._has_channel_dimension(inputs):
            return inputs, True

        return [inputs], False

    def _zeros_like(self, matrix):
        return [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

    def _flatten_matrix(self, matrix):
        return [value for row in matrix for value in row]

    def _add_in_place(self, baseMatrix, extraMatrix):
        for rowIndex in range(len(baseMatrix)):
            for columnIndex in range(len(baseMatrix[rowIndex])):
                baseMatrix[rowIndex][columnIndex] += extraMatrix[rowIndex][columnIndex]

    def _add_bias(self, matrix, bias):
        return [[value + bias for value in row] for row in matrix]

    def _flip_weights(self, weights):
        return [weightsRow[::-1] for weightsRow in weights[::-1]]

    def _initialize_weights(self, in_channels):
        if self.fixedKernels is not None or len(self.weights) > 0:
            return

        self.in_channels = in_channels
        self.inputsCount = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weightsCount = self.inputsCount + 1

        randomInstance = random.Random(self.seed)
        self.weights = []

        for _ in range(self.out_channels):
            flatKernelWeights = []

            for _ in range(self.in_channels):
                for _ in range(self.kernel_size[0] * self.kernel_size[1]):
                    flatKernelWeights.append(randomInstance.uniform(-0.1, 0.1))

            self.weights.append(flatKernelWeights + [0.0])

    def _reconstruct_kernel(self, weightsVector, inputChannelIndex):
        kernelHeight, kernelWidth = self.kernel_size
        kernelValuesCount = kernelHeight * kernelWidth
        startIndex = inputChannelIndex * kernelValuesCount
        endIndex = startIndex + kernelValuesCount
        kernelFlat = weightsVector[startIndex:endIndex]

        return [
            kernelFlat[rowIndex * kernelWidth:(rowIndex + 1) * kernelWidth]
            for rowIndex in range(kernelHeight)
        ]

    def _convolve_single(self, inputChannel, kernel, bias=0.0):
        conv = Convolute2DFunction(kernel, self.stride, self.dilation, self.padding)
        output = conv.call(inputChannel)
        return self._add_bias(output, bias)

    def _forward_single_fixed(self, inputs):
        inputChannels, hadChannelDimension = self._normalize_channels(inputs)

        if len(self.fixedKernels) == 1 and len(inputChannels) > 1:
            return [self._convolve_single(singleChannel, self.fixedKernels[0]) for singleChannel in inputChannels]

        if len(inputChannels) == 1 and len(self.fixedKernels) == 1:
            output = self._convolve_single(inputChannels[0], self.fixedKernels[0])

            if hadChannelDimension:
                return [output]

            return output

        outputs = []

        if len(inputChannels) == 1:
            for singleKernel in self.fixedKernels:
                outputs.append(self._convolve_single(inputChannels[0], singleKernel))
        else:
            for channelIndex in range(min(len(inputChannels), len(self.fixedKernels))):
                outputs.append(self._convolve_single(inputChannels[channelIndex], self.fixedKernels[channelIndex]))

        if not hadChannelDimension and len(outputs) == 1:
            return outputs[0]

        return outputs

    def _forward_single_trainable(self, inputs):
        inputChannels, hadChannelDimension = self._normalize_channels(inputs)
        self._initialize_weights(len(inputChannels))

        outputs = []

        for outputChannelIndex in range(self.out_channels):
            channelOutput = None

            for inputChannelIndex in range(self.in_channels):
                kernel = self._reconstruct_kernel(self.weights[outputChannelIndex][:-1], inputChannelIndex)
                currentOutput = Convolute2DFunction(kernel, self.stride, self.dilation, self.padding).call(inputChannels[inputChannelIndex])

                if channelOutput is None:
                    channelOutput = currentOutput
                else:
                    self._add_in_place(channelOutput, currentOutput)

            outputs.append(self._add_bias(channelOutput, self.weights[outputChannelIndex][-1]))

        if not hadChannelDimension and len(outputs) == 1:
            return outputs[0]

        return outputs

    def forwardPass(self, inputs, debug=False):
        _ = debug

        if self._is_batch_input(inputs):
            return [self.forwardPass(singleInput) for singleInput in inputs]

        if self.fixedKernels is not None:
            return self._forward_single_fixed(inputs)

        return self._forward_single_trainable(inputs)

    def _compute_weight_gradient(self, inputMatrix, outputGradient, kernelHeight, kernelWidth):
        gradient = [[0 for _ in range(kernelWidth)] for _ in range(kernelHeight)]

        for kernelRowIndex in range(kernelHeight):
            for kernelColumnIndex in range(kernelWidth):
                for outputRowIndex in range(len(outputGradient)):
                    for outputColumnIndex in range(len(outputGradient[0])):
                        inputRowIndex = outputRowIndex + kernelRowIndex
                        inputColumnIndex = outputColumnIndex + kernelColumnIndex
                        gradient[kernelRowIndex][kernelColumnIndex] += (
                            inputMatrix[inputRowIndex][inputColumnIndex] *
                            outputGradient[outputRowIndex][outputColumnIndex]
                        )

        return gradient

    def _compute_input_gradient(self, outputGradient, kernel, inputShape):
        inputHeight, inputWidth = inputShape
        flippedKernel = self._flip_weights(kernel)
        gradient = [[0 for _ in range(inputWidth)] for _ in range(inputHeight)]

        for outputRowIndex in range(len(outputGradient)):
            for outputColumnIndex in range(len(outputGradient[0])):
                for kernelRowIndex in range(len(flippedKernel)):
                    for kernelColumnIndex in range(len(flippedKernel[0])):
                        inputRowIndex = outputRowIndex + kernelRowIndex
                        inputColumnIndex = outputColumnIndex + kernelColumnIndex

                        if inputRowIndex < inputHeight and inputColumnIndex < inputWidth:
                            gradient[inputRowIndex][inputColumnIndex] += (
                                outputGradient[outputRowIndex][outputColumnIndex] *
                                flippedKernel[kernelRowIndex][kernelColumnIndex]
                            )

        return gradient

    def backwardPass(self, previousLayerOutputs, expectedOutputsErrorDerivatives=None, learningRate=0.01, debug=False, **kwargs):
        _ = debug

        legacyMode = "learning_rate" in kwargs
        if legacyMode:
            learningRate = kwargs["learning_rate"]
            previousLayerOutputs, expectedOutputsErrorDerivatives = expectedOutputsErrorDerivatives, previousLayerOutputs

        previousChannels, previousHadChannelDimension = self._normalize_channels(previousLayerOutputs)
        outputChannels, _ = self._normalize_channels(expectedOutputsErrorDerivatives)

        if self.fixedKernels is not None:
            sharedKernel = self.fixedKernels[0]
            accumulatedKernelGradient = [[0 for _ in range(len(sharedKernel[0]))] for _ in range(len(sharedKernel))]
            nextLayerErrorDerivatives = []

            for channelIndex in range(len(outputChannels)):
                currentInput = previousChannels[channelIndex if len(previousChannels) > 1 else 0]
                currentGradient = outputChannels[channelIndex]
                kernelGradient = self._compute_weight_gradient(
                    currentInput,
                    currentGradient,
                    len(sharedKernel),
                    len(sharedKernel[0]),
                )
                self._add_in_place(accumulatedKernelGradient, kernelGradient)
                nextLayerErrorDerivatives.append(
                    self._compute_input_gradient(
                        currentGradient,
                        sharedKernel,
                        (len(currentInput), len(currentInput[0])),
                    )
                )

            self.kernel = [
                [
                    self.kernel[rowIndex][columnIndex] - learningRate * accumulatedKernelGradient[rowIndex][columnIndex]
                    for columnIndex in range(len(self.kernel[rowIndex]))
                ]
                for rowIndex in range(len(self.kernel))
            ]
            self.fixedKernels = [self.kernel]

            if not previousHadChannelDimension and len(nextLayerErrorDerivatives) == 1:
                nextLayerErrorDerivatives = nextLayerErrorDerivatives[0]

            if legacyMode:
                return nextLayerErrorDerivatives

            flatKernelGradient = self._flatten_matrix(accumulatedKernelGradient)
            return flatKernelGradient + [0.0], nextLayerErrorDerivatives

        self._initialize_weights(len(previousChannels))
        layerWeightsDerivativesVector = []
        nextLayerErrorDerivatives = [
            self._zeros_like(previousChannels[inputChannelIndex])
            for inputChannelIndex in range(len(previousChannels))
        ]

        for outputChannelIndex in range(self.out_channels):
            outputGradient = outputChannels[outputChannelIndex]
            outputWeights = self.weights[outputChannelIndex]

            for inputChannelIndex in range(self.in_channels):
                kernel = self._reconstruct_kernel(outputWeights[:-1], inputChannelIndex)
                kernelGradient = self._compute_weight_gradient(
                    previousChannels[inputChannelIndex],
                    outputGradient,
                    self.kernel_size[0],
                    self.kernel_size[1],
                )

                layerWeightsDerivativesVector.extend(self._flatten_matrix(kernelGradient))

                inputGradient = self._compute_input_gradient(
                    outputGradient,
                    kernel,
                    (len(previousChannels[inputChannelIndex]), len(previousChannels[inputChannelIndex][0])),
                )
                self._add_in_place(nextLayerErrorDerivatives[inputChannelIndex], inputGradient)

            biasGradient = sum(sum(row) for row in outputGradient)
            layerWeightsDerivativesVector.append(biasGradient)

        if not previousHadChannelDimension and len(nextLayerErrorDerivatives) == 1:
            nextLayerErrorDerivatives = nextLayerErrorDerivatives[0]

        if legacyMode:
            return nextLayerErrorDerivatives

        return layerWeightsDerivativesVector, nextLayerErrorDerivatives
