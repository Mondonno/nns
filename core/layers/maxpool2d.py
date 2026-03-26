from nns.core.layers.layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, poolSize=(2, 2), stride=None):
        super().__init__()
        self.poolSize = poolSize
        self.stride = stride if stride is not None else poolSize
        self.lastInputShape = None
        self.lastMaxIndices = None
        self.lastHadChannelDimension = False

    def forwardPass(self, inputs):
        if isinstance(inputs[0][0], list):
            self.lastHadChannelDimension = True
            self.lastInputShape = [len(inputs), len(inputs[0]), len(inputs[0][0])]

            pooledOutputs = []
            maxIndices = []

            for channel in inputs:
                pooledOutput, channelIndices = self._poolSingleChannel(channel)
                pooledOutputs.append(pooledOutput)
                maxIndices.append(channelIndices)

            self.lastMaxIndices = maxIndices
            return pooledOutputs

        self.lastHadChannelDimension = False
        self.lastInputShape = [len(inputs), len(inputs[0])]
        pooledOutput, maxIndices = self._poolSingleChannel(inputs)
        self.lastMaxIndices = maxIndices
        return pooledOutput

    def _poolSingleChannel(self, inputChannel):
        inputHeight, inputWidth = len(inputChannel), len(inputChannel[0])
        poolHeight, poolWidth = self.poolSize
        strideHeight, strideWidth = self.stride
        outputHeight = (inputHeight - poolHeight) // strideHeight + 1
        outputWidth = (inputWidth - poolWidth) // strideWidth + 1
        outputMatrix = [[0 for _ in range(outputWidth)] for _ in range(outputHeight)]
        maxIndices = [[(0, 0) for _ in range(outputWidth)] for _ in range(outputHeight)]
        for outputRowIndex in range(outputHeight):
            for outputColIndex in range(outputWidth):
                window = [row[outputColIndex*strideWidth:outputColIndex*strideWidth+poolWidth] for row in inputChannel[outputRowIndex*strideHeight:outputRowIndex*strideHeight+poolHeight]]
                flatWindow = sum(window, [])
                maxValue = max(flatWindow)
                maxIndex = flatWindow.index(maxValue)
                maxIndices[outputRowIndex][outputColIndex] = (outputRowIndex*strideHeight + maxIndex // poolWidth, outputColIndex*strideWidth + maxIndex % poolWidth)
                outputMatrix[outputRowIndex][outputColIndex] = maxValue
        return outputMatrix, maxIndices

    def backwardPass(self, previousLayerOutputs, expectedOutputsErrorDerivatives=None):
        _ = previousLayerOutputs

        if expectedOutputsErrorDerivatives is None:
            dL_dout = previousLayerOutputs
        else:
            dL_dout = expectedOutputsErrorDerivatives

        if self.lastInputShape is None or self.lastMaxIndices is None:
            raise ValueError("Must call forwardPass before backwardPass")

        if self.lastHadChannelDimension:
            channelCount, inputHeight, inputWidth = self.lastInputShape
            gradientWrtInput = [[[0 for _ in range(inputWidth)] for _ in range(inputHeight)] for _ in range(channelCount)]
            for channelIndex in range(channelCount):
                outputHeight = len(dL_dout[channelIndex])
                outputWidth = len(dL_dout[channelIndex][0])
                for outputRowIndex in range(outputHeight):
                    for outputColIndex in range(outputWidth):
                        maxRowIndex, maxColIndex = self.lastMaxIndices[channelIndex][outputRowIndex][outputColIndex]
                        gradientWrtInput[channelIndex][maxRowIndex][maxColIndex] = dL_dout[channelIndex][outputRowIndex][outputColIndex]
            return gradientWrtInput

        inputHeight, inputWidth = self.lastInputShape
        gradientWrtInput = [[0 for _ in range(inputWidth)] for _ in range(inputHeight)]
        outputHeight = len(dL_dout)
        outputWidth = len(dL_dout[0])
        for outputRowIndex in range(outputHeight):
            for outputColIndex in range(outputWidth):
                maxRowIndex, maxColIndex = self.lastMaxIndices[outputRowIndex][outputColIndex]
                gradientWrtInput[maxRowIndex][maxColIndex] = dL_dout[outputRowIndex][outputColIndex]
        return gradientWrtInput
