from ..layers.dict import layersDict
from ..functions.dict import functionsDict

from .model import Model
from ..layers import Layer
from ..functions.function import Function

class Sequential(Model):
    def __init__(self, layers, error, learningRate, optimizers = None):
        super().__init__()

        self.name = self.__class__.__name__

        self.layers: list[Layer] = layers
        # adding to the layers error 

        self.error: Function = error

        self.optimizers = optimizers if optimizers is not None else []

        self.learningRate: Function = learningRate

    def fit(self, *args, **kwargs):
        gradientTape = self.fitWithGradientTape(*args, **kwargs)

        _ = list(gradientTape)

    def fitWithGradientTape(self, dataset, epochs = 1):
        for epochIndex in range(epochs):
            epochErrorValues = []

            datasetBatches = dataset.generate()

            # 600 / 4 batchy
            for batchIndex in range(len(datasetBatches)):
                batch = datasetBatches[batchIndex]

                weightsDerivativesVector = []

                # del C / del w
                # weightsDerivativesMatrix = [ [ [ [] for _ in range(len(self.layers[i].weights[j])) ] for j in range(self.layers[i].neuronsCount)] for i in range(len(self.layers)) ]                
                batchErrorValues = []

                for batchInputs, batchOutputs in batch:
                    layerOutputs = self.forwardPass(batchInputs)
                    

                    # lastLayerIndex = len(self.layers) - 1

                    # for i in range(len(layerOutputs[lastLayerIndex])):     
                    #     for j in range(len(layerOutputs[lastLayerIndex][i])):
                    #         errorDerivativesForNeurons[lastLayerIndex][i] = self.error.derivative((layerOutputs[lastLayerIndex][i][j], batchOutputs[i][j]))
                    
                    # activationDerivativesForNeurons[lastLayerIndex] = self.layers[lastLayerIndex].derivatives(layerOutputs[lastLayerIndex])

                    # now going on with backpropagation algorithm (the whole fit is a backprop but here we are calculating the weights derviatives)
                    # given we have n layers we n'th layer is at the end
                    # we start with n'th layer so previousLayer is (n - 1)'th layer so it means it is the earlier layer when we will visualize the network (if we go backwards it is the forward layer)
                    # we can say we are looking from the end
                    # 1 2 3
                    # .
                    # . . .
                    # . . .
                    # .
                    #     ^
                    #     | we look from here so previous is layer no. 2 (i)
                    # dots are neurons

                    currentLayerErrorDerivatives = []

                    for i in reversed(range(1, len(layerOutputs))):
                        # layerOutputs[i-1] are the outputs of the layer to input to the L'th layer of network (these are outputs of (L-1)'th layer)
                        # (i - 1) index is the index of the L'th layer

                        layerWeightsDerivativesVector = []

                        previousLayerOutputsIndex = i - 1
                        previousLayerOutputs = layerOutputs[previousLayerOutputsIndex] # TODO: naming to change since it is quite confusing (this is input for the current layer for backward pass method)

                        currentLayerOutputsIndex = i
                        currentLayerOutputs = layerOutputs[currentLayerOutputsIndex]

                        previousLayerIndex = i - 2

                        currentLayerIndex = i - 1
                        currentLayer = self.layers[currentLayerIndex]

                        currentLayerNeuronsCount = getattr(currentLayer, "neuronsCount", len(currentLayerOutputs))

                        if(len(currentLayerErrorDerivatives) == 0):
                            currentLayerErrorDerivatives = [ None for _ in range(currentLayerNeuronsCount)]

                        if((currentLayerIndex + 1) >= len(self.layers)): # we are on the last layer if the condition passes
                            for k in range(currentLayer.neuronsCount): # a connection (input and weight) between the k'th and m'th neuron, respectively from layer L, and L + 1 ( the m'th neuron on the L + 1 layer has the weight )
                                # print(currentLayerOutputs)
                                # outputForNeuron = currentLayerOutputs[k][0]
                                outputForNeuron = currentLayerOutputs[k][0]
                                exceptedOutputForNeuron = batchOutputs[k]
                                # print("LASY LAYER ERROR")
                                # print(outputForNeuron, exceptedOutputForNeuron)

                                errorDerivative = self.error.derivative((outputForNeuron, exceptedOutputForNeuron))

                                # print("ERD Standard", errorDerivative)

                                # if errorDerivative > 1e2:
                                #     print("Batch data at overcome", batchIndex, epochIndex, currentLayerIndex, "One TIME")
                                #     print("Overcoming the error derivative!", errorDerivative)

                                # print("Error derivative data: ", outputForNeuron, exceptedOutputForNeuron, errorDerivative)

                                batchErrorValues.append(self.error.call((outputForNeuron, exceptedOutputForNeuron)))

                                # * activationDerivativesForNeurons[currentLayerIndex][k]

                                # del C / del z
                                currentLayerErrorDerivatives[k] = errorDerivative

                        backwardPassResult = currentLayer.backwardPass(previousLayerOutputs, currentLayerErrorDerivatives)

                        if isinstance(backwardPassResult, tuple):
                            layerWeightsDerivativesVector, nextLayerErrorDerivatives = backwardPassResult
                        else:
                            layerWeightsDerivativesVector = []
                            nextLayerErrorDerivatives = backwardPassResult

                        currentLayerErrorDerivatives = nextLayerErrorDerivatives

                        # it is append at the beggining 
                        if len(layerWeightsDerivativesVector) > 0:
                            weightsDerivativesVector[:0] = (layerWeightsDerivativesVector)
                        # activationDerivativesForNeurons[currentLayerIndex] = self.layers[currentLayerIndex].derivatives(layerOutputs[i])
                    
                    weightsDerivativesVectorIndex = 0

                    for i in range(len(self.layers)):
                        if not hasattr(self.layers[i], "weights"):
                            continue

                        for k in range(len(self.layers[i].weights)):
                            for l in range(len(self.layers[i].weights[k])):
                                # weightsDerivativesSum = sum(weightsDerivativesMatrix[i][k][l])
                                # weightsDerivativesCount = len(weightsDerivativesMatrix[i][k][l])

                                # if(weightsDerivativesCount == 0):
                                #     raise TypeError()
                                
                                # singleGradientMatrixValue = weightsDerivativesSum / weightsDerivativesCount

                                singleGradientValue = weightsDerivativesVector[weightsDerivativesVectorIndex]

                                for singleOptimizer in self.optimizers:
                                    singleGradientValue = singleOptimizer.call(singleGradientValue, weightsDerivativesVector)

                                yield singleGradientValue

                                learningRateValue = self.learningRate.call(epochIndex)

                                self.layers[i].weights[k][l] -= singleGradientValue * learningRateValue

                                weightsDerivativesVectorIndex += 1

                epochErrorValues.append(sum(batchErrorValues) / len(batchErrorValues))
            
            epochMeanError = sum(epochErrorValues) / len(epochErrorValues)

            print("Trained: ", epochIndex, "Error: ", epochMeanError)

    def forwardPass(self, inputs):
        outputs = [  [ inputs ]  ]

        for i in range(1, len(self.layers) + 1):
            currentLayerOutputsIndex = i - 1
            currentLayerOutputs = outputs[i - 1]
            currentLayerForwardResult = self.layers[currentLayerOutputsIndex].forwardPass(currentLayerOutputs)

            if isinstance(currentLayerForwardResult, tuple):
                currentLayerInputs, currentLayerOutputs = currentLayerForwardResult
            else:
                currentLayerInputs = currentLayerOutputs
                currentLayerOutputs = currentLayerForwardResult

            embeddedCurrentLayerOutputs = currentLayerOutputs

            outputs.append(embeddedCurrentLayerOutputs)
            outputs[i - 1] = currentLayerInputs

        return outputs

    def forwardPassByOutputLayer(self, inputs):
        outputs = self.forwardPass(inputs)

        return outputs[len(outputs) - 1]
    
    @classmethod
    def fromDict(self, objectDict, additionalDict):
        referenceDict = (layersDict | functionsDict) | additionalDict

        learningRateObject = objectDict["learningRate"]
        learningRateTypeName = learningRateObject["name"]
        learningRateType = referenceDict[learningRateTypeName]

        learningRate = learningRateType.fromDict(learningRateObject, referenceDict)

        errorObject = objectDict["error"]
        errorTypeName = errorObject["name"]
        errorType = referenceDict[errorTypeName]

        error = errorType.fromDict(errorObject, referenceDict)

        layers = []

        for singleLayer in objectDict["layers"]:
            layerTypeName = singleLayer["name"]
            layerType = referenceDict[layerTypeName]
            
            layers.append(layerType.fromDict(singleLayer, referenceDict))

        return Sequential(layers, error, learningRate)
