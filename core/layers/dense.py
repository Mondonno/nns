import math

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
            
            # a bias
            singleWeightedSum += self.weights[i][self._inputsCount]

            weightedSum[i] = singleWeightedSum
                
        return weightedSum

    def derivatives(self, inputs):
        # del a / del z

        filledInputs = self.__fillInputs(inputs)

        # print("IIII: ",inputs, self._inputsCount, self._neuronsCount)
        # print("FFFF: ",filledInputs)

        weightedSum = self.weightedSum(filledInputs)
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