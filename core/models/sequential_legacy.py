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
                weightsDerivativesMatrix = [ [ [ [] for _ in range(len(self.layers[i].weights[j])) ] for j in range(self.layers[i].neuronsCount)] for i in range(len(self.layers)) ]                
                batchErrorValues = []

                for batchInputs, batchOutputs in batch:
                    # del C / del a
                    errorDerivativesForNeurons = [ [ None for _ in range(self.layers[i].neuronsCount)] for i in range(len(self.layers)) ]

                    # del a / del z
                    activationDerivativesForNeurons = [ [ None for _ in range(self.layers[i].neuronsCount) ] for i in range(len(self.layers)) ]

                    # TODO: create arrays for: del a2/ del z1

                    # print("Inputs and outputs", batchInputs, batchOutputs)
                    # print("Weights", self.layers[0].weights)

                    # print("Inputs", batchInputs)

                    layerOutputs = self.forwardPass(batchInputs)

                    # print(layerOutputs)

                    for i in range(1, len(layerOutputs)):
                        currentLayerIndex = i - 1
                        currentLayerOutputIndex = i - 1

                        # print("Current layer for deri", currentLayerIndex)

                        # print("Deri", layerOutputs[i])

                        activationDerivativesForNeurons[currentLayerIndex] = self.layers[currentLayerIndex].derivatives(layerOutputs[currentLayerOutputIndex])

                    # print("Layer outputs", layerOutputs)

                    # lastLayerIndex = len(self.layers) - 1

                    # for i in range(len(layerOutputs[lastLayerIndex])):     
                    #     for j in range(len(layerOutputs[lastLayerIndex][i])):
                    #         errorDerivativesForNeurons[lastLayerIndex][i] = self.error.derivative((layerOutputs[lastLayerIndex][i][j], batchOutputs[i][j]))

                    # activationDerivativesForNeurons[lastLayerIndex] = self.layers[lastLayerIndex].derivatives(layerOutputs[lastLayerIndex])

                    # now going on with backpropagation algorithm (the whole fit is a backprop but here we are calculating the weights derviatives)
                    for i in reversed(range(1, len(layerOutputs))):
                        # layerOutputs[i-1] are the outputs of the layer to input to the L'th layer of network (these are outputs of (L-1)'th layer)
                        # (i - 1) index is the index of the L'th layer

                        layerWeightsDerivativesVector = []

                        previousLayerOutputsIndex = i - 1
                        previousLayerOutputs = layerOutputs[previousLayerOutputsIndex]

                        currentLayerOutputsIndex = i
                        currentLayerOutputs = layerOutputs[currentLayerOutputsIndex]

                        previousLayerIndex = i - 2

                        currentLayerIndex = i - 1
                        currentLayer = self.layers[currentLayerIndex]

                        for k in range(currentLayer.neuronsCount): # a connection (input and weight) between the k'th and m'th neuron, respectively from layer L, and L + 1 ( the m'th neuron on the L + 1 layer has the weight )
                            currentLayerWeights = currentLayer.weights[k]
                            currentLayerInputsCount = currentLayer.inputsCount
                            currentLayerActualWeightsCount = currentLayer.weightsCount

                            if((currentLayerIndex + 1) >= len(self.layers)): # we are on the last layer if the condition passes

                                # print(currentLayerOutputs)
                                outputForNeuron = currentLayerOutputs[k][0]
                                exceptedOutputForNeuron = batchOutputs[k]
                                # print("LASY LAYER ERROR")
                                # print(outputForNeuron, exceptedOutputForNeuron)

                                errorDerivative = self.error.derivative((outputForNeuron, exceptedOutputForNeuron))

                                # print("ERD Standard", errorDerivative)

                                # if errorDerivative > 1e2:
                                #     print("Batch data at overcome", batchIndex, epochIndex, currentLayerIndex, "One TIME")
                                #     print("Overcoming the error derivative!", errorDerivative)

                                batchErrorValues.append(self.error.call((outputForNeuron, exceptedOutputForNeuron)))

                                # * activationDerivativesForNeurons[currentLayerIndex][k]

                                # del C / del z 
                                errorDerivativesForNeurons[currentLayerIndex][k] = errorDerivative

                            # print("M loops", currentLayerInputsCount)

                            for m in range(currentLayerInputsCount): # the len(previousLayerOutputs) of previous layer outputs should be equal to the inputs count of current layer
                                # print(currentErrorDerivative)

                                # print(currentLayerIndex)
                                if(previousLayerIndex < 0):
                                    break

                                currentWeight = currentLayerWeights[m]
                                # currentInput = previousLayerOutputs[k][m]

                                # del a / del z
                                currentActivationDerivative = activationDerivativesForNeurons[currentLayerIndex][k] # m'th neuron from L Layer

                                # [del C / del a]
                                currentLayerErrorDerivatives = errorDerivativesForNeurons[currentLayerIndex]

                                # del C / del z
                                currentErrorDerivative = currentLayerErrorDerivatives[k] or 1

                                # print("   : ", currentWeight, currentActivationDerivative, currentErrorDerivative)
                                # print(errorDerivativesForNeurons, previousLayerIndex, currentLayerIndex, m, k)

                                if (errorDerivativesForNeurons[previousLayerIndex][m] == None):
                                    # print("change")
                                    errorDerivativesForNeurons[previousLayerIndex][m] = 0

                                # del C / del z = del a / del z * del C / del a
                                # del C / del z
                                # TODO: remove this is not needed AFAIK
                                errorDerivativesForNeurons[previousLayerIndex][m] += currentWeight * currentActivationDerivative * currentErrorDerivative
                                # print(errorDerivativesForNeurons[previousLayerIndex][m], currentWeight, currentActivationDerivative, currentErrorDerivative)

                            # print(errorDerivativesForNeurons)
                            for l in range(currentLayerActualWeightsCount):
                                currentPreviousLayerOutput = 1

                                if(l != currentLayerInputsCount): # handling bias
                                    # print(k, l, previousLayerOutputsIndex)
                                    # print(k, l, previousLayerOutputs, previousLayerOutputs[k])
                                    currentPreviousLayerOutput = previousLayerOutputs[k][l]

                                currentActivationDerivative = activationDerivativesForNeurons[currentLayerIndex][k]
                                currentErrorDerivative = errorDerivativesForNeurons[currentLayerIndex][k] or 1

                                singleWeightDerivative = currentPreviousLayerOutput * currentActivationDerivative * currentErrorDerivative

                                if(singleWeightDerivative != 0):
                                    # raise TypeError()
                                    pass

                                # print(currentLayerIndex, k, l, singleWeightDerivative)

                                layerWeightsDerivativesVector.append(singleWeightDerivative)
                                weightsDerivativesMatrix[currentLayerIndex][k][l].append(singleWeightDerivative)


                        self.layers[i].backwardPass()

                        weightsDerivativesVector[:0] = (layerWeightsDerivativesVector)
                        # activationDerivativesForNeurons[currentLayerIndex] = self.layers[currentLayerIndex].derivatives(layerOutputs[i])

                    weightsDerivativesVectorIndex = 0

                    for i in range(len(self.layers)):
                        for k in range(self.layers[i].neuronsCount):
                            for l in range(self.layers[i].inputsCount + 1):
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
            currentLayerInputs, currentLayerOutputs = self.layers[currentLayerOutputsIndex].forwardPass(currentLayerOutputs)

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