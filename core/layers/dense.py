import math
import random

from .initializators.dict import initializatorsDict
from .initializators.xavier import XavierInitializatorFunction

from .layer import Layer

class Dense(Layer):
    def __init__(self, inputsCount, neuronsCount, activation, seed = None, initializator = None, weights = None):
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

        self.initializator = XavierInitializatorFunction(seed=self.seed) if initializator is None else initializator

        # randomInstance = Random(self.seed)

        # x * r - 1 -> [-2, 2]

        # (2 * randomInstance.random() - 1)

        # ( (0 if i == self._inputsCount else 1) if (seed != None and math.isnan(seed)) else
        self.weights = [ [ (self.initializator.call(self, i, j))  for j in range(0, self._inputsCount + 1) ] for i in range(0, self._neuronsCount)] if weights is None else weights

        self.activation = activation

    def __flattenInputs(self, inputs):
        flattenedInputs = []
        for i in range(len(inputs)):
            if isinstance(inputs[i], list):
                for j in range(len(inputs[i])):
                    flattenedInputs.append(inputs[i][j])
            else:
                flattenedInputs.append(inputs[i])

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
        filledInputs = self.__fillInputs(inputs)
        weightedSum = self.weightedSum(filledInputs)

        derivatives = []

        for i in range(self._neuronsCount):
            derivatives.append(self.activation.derivative(weightedSum[i]))

        return derivatives

    def _normalizeErrorDerivatives(self, errorDerivatives):
        normalizedDerivatives = []

        for singleDerivative in errorDerivatives:
            if isinstance(singleDerivative, list):
                normalizedDerivatives.append(singleDerivative[0])
            else:
                normalizedDerivatives.append(singleDerivative)

        return normalizedDerivatives

    def _computeCurrentLayerDeltas(self, weightedSums, currentLayerErrorDerivatives):
        normalizedErrorDerivatives = self._normalizeErrorDerivatives(currentLayerErrorDerivatives)

        if hasattr(self.activation, "jacobian"):
            jacobianMatrix = self.activation.jacobian(weightedSums)
            currentLayerDeltas = []

            for columnIndex in range(len(jacobianMatrix[0])):
                singleDelta = 0

                for rowIndex in range(len(jacobianMatrix)):
                    singleDelta += jacobianMatrix[rowIndex][columnIndex] * normalizedErrorDerivatives[rowIndex]

                currentLayerDeltas.append(singleDelta)

            return currentLayerDeltas

        return [
            self.activation.derivative(weightedSums[index]) * normalizedErrorDerivatives[index]
            for index in range(len(weightedSums))
        ]
    
    def forwardPass(self, inputs, debug= False):
        if debug:
            print("Inputs: ", inputs)

        filledInputs = self.__fillInputs(inputs)
        weightedSum = self.weightedSum(filledInputs, debug)

        for singleWeightedSum in weightedSum:
            if(singleWeightedSum > 1e250):
                print("Overreaching! ", singleWeightedSum)

            if(math.isnan(singleWeightedSum)):
                raise TypeError()

        if hasattr(self.activation, "callVector"):
            activatedSums = self.activation.callVector(weightedSum)
            outputs = [[activatedSum] for activatedSum in activatedSums]
            return filledInputs, outputs

        outputs = []

        for i in range(self._neuronsCount):
            activatedSum = self.activation.call(weightedSum[i])
            outputs.append([activatedSum])
        
        return filledInputs, outputs
    
    def backwardPass(self, inputs, expectedOutputsErrorDerivatives, debug=False):
        filledInputs, _ = self.forwardPass(inputs)
        layerWeightsDerivativesVector = []

        currentLayer = self

        currentLayerInputsCount = currentLayer.inputsCount
        currentLayerActualWeightsCount = currentLayer.weightsCount 
        currentLayerInputsWithBiasCount = currentLayerActualWeightsCount
        currentLayerWeightedSums = self.weightedSum(filledInputs, debug)
        currentLayerDeltas = self._computeCurrentLayerDeltas(currentLayerWeightedSums, expectedOutputsErrorDerivatives)
        nextLayerErrorDerivatives = [0 for _ in range(currentLayerInputsCount)]

        for k in range(currentLayer.neuronsCount):
            currentLayerWeights = currentLayer.weights[k]

            for m in range(currentLayerInputsCount):
                currentWeight = currentLayerWeights[m]
                nextLayerErrorDerivatives[m] += currentWeight * currentLayerDeltas[k]

            for l in range(currentLayerInputsWithBiasCount):
                currentPreviousLayerOutput = 1

                if(l != currentLayerInputsCount):
                    currentPreviousLayerOutput = filledInputs[k][l]

                singleWeightDerivative = currentPreviousLayerOutput * currentLayerDeltas[k]

                layerWeightsDerivativesVector.append(singleWeightDerivative)

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
