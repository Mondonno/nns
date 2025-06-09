from nns.core.layers.layer import Layer

class MaxPooling2D(Layer):
    def __init__(self, poolSize=(2, 2), stride=None):
        super().__init__()
        self.poolSize = poolSize
        self.stride = stride if stride is not None else poolSize
        self.lastInputShape = None
        self.lastMaxIndices = None

    def forwardPass(self, inputs):
        # inputs: [channel][height][width] or [height][width]
        self.lastInputShape = [len(inputs)] + ([len(inputs[0]), len(inputs[0][0])] if isinstance(inputs[0][0], list) else [len(inputs[0])]) if isinstance(inputs[0], list) else [len(inputs)]
        if isinstance(inputs[0][0], list):  # multi-channel
            return [self._poolSingleChannel(channel) for channel in inputs]
        else:
            return self._poolSingleChannel(inputs)

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
        self.lastMaxIndices = maxIndices
        return outputMatrix

    def backwardPass(self, dL_dout):
        # dL_dout: gradient from next layer, shape matches output of forwardPass
        if self.lastInputShape is None or self.lastMaxIndices is None:
            raise ValueError("Must call forwardPass before backwardPass")
        if len(self.lastInputShape) == 3:  # multi-channel
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
        else:  # single channel
            inputHeight, inputWidth = self.lastInputShape
            gradientWrtInput = [[0 for _ in range(inputWidth)] for _ in range(inputHeight)]
            outputHeight = len(dL_dout)
            outputWidth = len(dL_dout[0])
            for outputRowIndex in range(outputHeight):
                for outputColIndex in range(outputWidth):
                    maxRowIndex, maxColIndex = self.lastMaxIndices[outputRowIndex][outputColIndex]
                    gradientWrtInput[maxRowIndex][maxColIndex] = dL_dout[outputRowIndex][outputColIndex]
            return gradientWrtInput
