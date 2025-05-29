import math
import random

from .initializators.dict import initializatorsDict
from .initializators.xavier import XavierInitializatorFunction

from .layer import Layer

class Dense(Layer):
    def __init__(self, inputsCount, neuronsCount, activation, seed, initializator = None, weights = None):
        super().__init__()

        self.name = self.__class__.__name__

        self._neuronsCount = neuronsCount
        self._inputsCount = inputsCount
        self._weightsCount = self._inputsCount + 1

        self.neuronsCount = self._neuronsCount
        self.inputsCount = self._inputsCount
        self.weightsCount = self._weightsCount

        if seed is None:
            self.seed = random.Random(seed).random()
        else:
            self.seed = seed

        self.initializator = XavierInitializatorFunction() if initializator is None else initializator

        # randomInstance = Random(self.seed)

        # x * r - 1 -> [-2, 2]

        # (2 * randomInstance.random() - 1)

        # ( (0 if i == self._inputsCount else 1) if (seed != None and math.isnan(seed)) else
        self.weights = [ [ (self.initializator.call(self, i, j))  for j in range(0, self._inputsCount + 1) ] for i in range(0, self._neuronsCount)] if weights is None else weights

        self.activation = activation

    def __flattenInputs(self, inputs):
        flattenedInputs = []
        for i in range(len(inputs)):
            for j in range(len(inputs[i])):
                flattenedInputs.append(inputs[i][j]) 

        return flattenedInputs

    # 1 -> 2 1
    def __fillInputs(self, inputs):
        # this makes each neuron with its own inputs set "replicted" to fill it's hunger

        # print("pre-Fla", inputs)
        flattenedInputs = self.__flattenInputs(inputs)
        # print("FLa", flattenedInputs)
        filledInputs = [flattenedInputs for _ in range(self._neuronsCount)] if len(flattenedInputs) == self._inputsCount else inputs
        return filledInputs

    # the rows can be a symbol of one neuron inputs/weights
    
    def weightedSum(self, inputs, debug = False): 
        weightedSum = [ 0 for _ in range(0, self._neuronsCount) ]
        
        if debug is True:
            print("HPA", self._weightsCount, self._inputsCount, self._neuronsCount)
            print("HPV", self.weights)
            print("WSI", inputs)

        for i in range(0, self._neuronsCount):
            # each neuron
            # inputs[i]

            singleWeightedSum = 0

            for j in range(0, self._inputsCount):
                if debug is True:
                    print("INS", i, j)
                    _ = inputs[i][j]
                    _ = self.weights[i][j]
                    print(self.weights[i][j], inputs[i][j])

                singleWeightedSum += self.weights[i][j] * inputs[i][j]
            
            # a bias, it is the last on inputs
            singleWeightedSum += self.weights[i][self._inputsCount]

            weightedSum[i] = singleWeightedSum
                
        return weightedSum

    def derivatives(self, inputs):
        # del a / del z

        filledInputs = self.__fillInputs(inputs)

        # print("IIII: ",inputs, self._inputsCount, self._neuronsCount)
        # print("FFFF: ",filledInputs)

        weightedSum = self.weightedSum(filledInputs)

        # print(self.weights)
        # print(weightedSum)

        derivatives = []
        
        for i in range(self._neuronsCount):
            derivatives.append(self.activation.derivative(weightedSum[i]))
        
        return derivatives
    
    def forwardPass(self, inputs, debug= False):
        filledInputs = self.__fillInputs(inputs)

        # print(filledInputs, self._inputsCount, self._neuronsCount)
        weightedSum = self.weightedSum(filledInputs, debug)

        outputs = []
        
        for i in range(self._neuronsCount):
            if(weightedSum[i] > 1e250):
                print("Overreaching! ", weightedSum[i])

            if(math.isnan(weightedSum[i])):
                raise TypeError()
            
            # print(weightedSum[i])

            activatedSum = self.activation.call(weightedSum[i])
            outputs.append([activatedSum])
        
        return filledInputs, outputs
    
    def backwardPass(self, inputs, expectedOutputsErrorDerivatives, debug=False):
        
        #currentLayerInputs = previousLayerOutputs
        
        # _ = previousLayerOutputs # this is the inputs for the current layer 
        # _ = currentLayerOutputs # we calculate based on inputs 

        # filledInputs, layerOutputs = self.forwardPass(inputs)
        # 
        # ...
        # currentLayerActivationDerivativesForNeurons = currentLayer.derivatives(filledInputs)

        filledInputs, layerOutputs = self.forwardPass(inputs)
        layerWeightsDerivativesVector = []

        currentLayer = self
        currentLayerNeuronsCount = currentLayer.neuronsCount

        currentLayerInputsCount = currentLayer.inputsCount
        currentLayerActualWeightsCount = currentLayer.weightsCount 
        currentLayerInputsWithBiasCount = currentLayerActualWeightsCount

        # currentLayerInputsCount = self.layers[previousLayerIndex].neuronsCount
        # nextLayerErrorDerivatives = [ None for _ in range(self.layers[previousLayerIndex].neuronsCount)]
        # currentLayerErrorDerivatives = [ None for _ in range(currentLayerNeuronsCount)]
        # activationDerivativesForNeurons[currentLayerIndex] = self.layers[currentLayerIndex].derivatives(layerOutputs[currentLayerOutputIndex])
        currentLayerActivationDerivativesForNeurons = currentLayer.derivatives(filledInputs)

        currentLayerErrorDerivatives = expectedOutputsErrorDerivatives
        nextLayerErrorDerivatives = [ None for _ in range(currentLayerInputsCount)]

        for k in range(currentLayer.neuronsCount): # a connection (input and weight) between the k'th and m'th neuron, respectively from layer L, and L + 1 ( the m'th neuron on the L + 1 layer has the weight )
            currentLayerWeights = currentLayer.weights[k]

            # print("M loops", currentLayerInputsCount)

            # currentLayerInputsCount = self.layers[previousLayerIndex].neuronsCount
            for m in range(currentLayerInputsCount): # the len(previousLayerOutputs) of previous layer outputs should be equal to the inputs count of current layer
                # print(currentErrorDerivative)

                # print(currentLayerIndex)
                # if(previousLayerIndex < 0):
                #     break

                currentWeight = currentLayerWeights[m]
                # currentInput = previousLayerOutputs[k][m]

                # del a / del z
                # currentActivationDerivative = activationDerivativesForNeurons[currentLayerIndex][k] # m'th neuron from L Layer
                currentActivationDerivative = currentLayerActivationDerivativesForNeurons[k] # m'th neuron from L Layer

                # [del C / del a]
                # currentLayerErrorDerivatives = errorDerivativesForNeurons[currentLayerIndex]

                # del C / del z
                currentErrorDerivative = currentLayerErrorDerivatives[k] or 1

                if (nextLayerErrorDerivatives[m] == None):
                    # print("change")
                    nextLayerErrorDerivatives[m] = 0

                # del C / del z = del a / del z * del C / del a
                # del C / del z
                nextLayerErrorDerivatives[m] += currentWeight * currentActivationDerivative * currentErrorDerivative
                # print(errorDerivativesForNeurons[previousLayerIndex][m], currentWeight, currentActivationDerivative, currentErrorDerivative)

            # print(errorDerivativesForNeurons)
            for l in range(currentLayerInputsWithBiasCount):
                currentPreviousLayerOutput = 1

                if(l != currentLayerInputsCount): # handling bias
                    # print(k, l, previousLayerOutputsIndex)
                    # print(k, l, previousLayerOutputs, previousLayerOutputs[k])
                    # one neuron can have multiple inputs, so we need to get the k'th neuron and l'th output
                    currentPreviousLayerOutput = inputs[k][l]

                currentActivationDerivative = currentLayerActivationDerivativesForNeurons[k]
                currentErrorDerivative = currentLayerErrorDerivatives[k] or 1

                singleWeightDerivative = currentPreviousLayerOutput * currentActivationDerivative * currentErrorDerivative

                if(singleWeightDerivative != 0):
                    # raise TypeError()
                    pass

                # print(currentLayerIndex, k, l, singleWeightDerivative)

                layerWeightsDerivativesVector.append(singleWeightDerivative)
                # weightsDerivativesMatrix[currentLayerIndex][k][l].append(singleWeightDerivative)

        return layerWeightsDerivativesVector, nextLayerErrorDerivatives

    @classmethod
    def fromDict(self, objectDict, additionalDict):
        referenceDict = initializatorsDict | additionalDict

        inputsCount = objectDict["inputsCount"]
        neuronsCount = objectDict["neuronsCount"]
        
        activationObject = objectDict["activation"]
        activationTypeName = activationObject["name"]

        activationType = referenceDict[activationTypeName]
        activationInstance = activationType.fromDict(activationObject, referenceDict)

        initializatorObject = objectDict["initializator"]
        initializatorTypeName = initializatorObject["name"]

        initializatorType = referenceDict[initializatorTypeName]
        initializatorInstance = initializatorType.fromDict(initializatorObject, referenceDict)

        seed = objectDict["seed"]
        weights = objectDict["weights"]
    
        return Dense(inputsCount, neuronsCount, activationInstance, initializator=initializatorInstance, seed=seed, weights=weights)