import math
import sys
import matplotlib.pyplot as plot
from random import Random

class Function():
    def __init__(self):
        pass
    
    def call(self, input):
        raise Exception(f"Couldn't call Function with {input} without function")
    
    def derivative(self, input):
        raise Exception(f"Couldn't derivate Function with {input} without derivation")
    
class LinearFunction(Function):
    def __init__(self):
        super().__init__()
    
    def call(self, input):
        return input
    
    def derivative(self, _):
        return 1

class MSEFunction(Function):
    def __init__(self):
        super().__init__()
    
    def call(self, input):
        return math.pow(input[0] - input[1], 2)

    def derivative(self, input):
        return 2 * (input[0] - input[1])
    
class CrossEntropyFunction(Function):
    def __init__(self):
        super().__init__()
    
    def call(self, input):
        # ln'(x) = 1/x
        # -1/x * y
        # 
        return -math.log(input[0]) * input[1]
    
    # prediction, singleValue
    # s - y
    # y => E(x)
    # s => prediction
    def derivative(self, input):
        return -1 * 1/input[0] * input[1]
        
class SineFunction(Function):
    def __init__(self):
        super().__init__(lambda x : math.sin(x), lambda x : math.cos(x))

    def call(self, input):
        return math.sin(input)

    def derivative(self, input):
        return math.cos(input)

class SineSquareFunction(SineFunction):
    def __init__(self):
        super().__init__()
    
    def call(self, input):
        return super().call(input) ** 2
    
    def derivative(self, input):
        return 2 * super().call(input) * super().derivative(input)
    
class SigmoidFunction(Function):
    def __init__(self):
        super().__init__()
    
    def call(self, input):
        return 1 / (1 + math.pow(math.e, -input))
    
    def derivative(self, input):
        return self.call(input) * (1 - self.call(input))

class Perceptron():
    def __init__(self, inputsCount, seed, activation, error) -> None:
        self.inputsCount = inputsCount

        randomInstance = Random(seed)

        self.weights = [ randomInstance.random() for _ in range(self.inputsCount + 1) ] # the + 1 is for the bias
    
        self.activation: Function = activation
        self.error: Function = error

        self.learningRate = 0.1

    def train(self, dataset, epochs):
        errors = []
        for e in range(epochs):
            print("Epoch no: ", e)

            gradient, meanError = self.computeOneEpochGradientGivenDataset(dataset)
            errors.append(meanError)

            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - self.learningRate * gradient[i]
        
        return errors

    def computeOneEpochGradientGivenDataset(self, dataset):
        # calculating error and partial derivatives

        # elements in labels are lists with that amount of elements that the Perceptron has inputs
        labels = [ x[0] for x in dataset ]
        values = [ x[1] for x in dataset ]
        errors = []

        weightsDerivatives = [ [] for _ in range(len(self.weights)) ]

        errorDerivatives = []
        activationDerivatives = []

        for i in range(len(dataset)):
            singleLabel = labels[i]
            singleValue = values[i]

            prediction = self.predict(singleLabel)
            weightedSum = 0

            for i in range(self.inputsCount):
                weightedSum = weightedSum + self.weights[i] * singleLabel[i]

            weightedSum = weightedSum + self.weights[self.inputsCount]
            # predictionDeviate = prediction - singleValue

            singleError = self.error.call((prediction, singleValue)) # math.pow(predictionDeviate, 2)
            singleErrorDerivative = self.error.derivative((prediction, singleValue))
            singleActivationDerivative = self.activation.derivative(weightedSum)

            # print(singleActivationDerivative)

            # partial derivative using chain rule

            for i in range(self.inputsCount):
                weightDerivative = singleLabel[i] * singleActivationDerivative * singleErrorDerivative
                weightsDerivatives[i].append(weightDerivative)

            weightsDerivatives[self.inputsCount].append(singleActivationDerivative * singleErrorDerivative)

            errorDerivatives.append(singleErrorDerivative)
            activationDerivatives.append(singleActivationDerivative)

            errors.append(singleError)

        # calculating mean error

        meanError = sum(errors) / len(errors)

        print("Current mean error", meanError)

        # calculating partial derivatives means
        gradient = [[] for _ in range(len(self.weights))]

        for i in range(len(self.weights)):
            gradient[i] = sum(weightsDerivatives[i]) / len(weightsDerivatives[i])

        meanErrorDerivative = sum(errorDerivatives) / len(errorDerivatives)
        meanActivationDerivative = sum(activationDerivatives) / len(activationDerivatives)

        print(meanErrorDerivative, meanActivationDerivative, self.activation.derivative(0))

        return gradient, meanError

    def predict(self, inputs):
        weightedSum = 0

        for i in range(self.inputsCount):
            weightedSum = weightedSum + self.weights[i] * inputs[i]

        weightedSum = weightedSum + self.weights[self.inputsCount]
        return self.activation.call(weightedSum)

def makePredictionHoldFromPerceptron(perceptron):
    predictionOnHold = True
    predictionSessionIndex = 0

    while predictionOnHold: 
        print(f"Prediction session no. {predictionSessionIndex + 1}")

        inputs = [None for _ in range(perceptron.inputsCount)]

        print(f"Provide {perceptron.inputsCount} decimals")

        try:
            for i in range(perceptron.inputsCount):
                inputs[i] = float(input(f"{i}: "))
        
            print(f"Here's the result: {perceptron.predict(inputs)}")
        except:
            print("Gotta error, try again")
        
        print("Wanna continue? y/n")

        predictionOnHold = input()[0].lower() == "y"

def displayDatasetWithPerceptronPredictions(dataset, perceptron):
    labels = [ x[0] for x in dataset ]
    values = [ x[1] for x in dataset ]

    perdictedValues = [ perceptron.predict(x[0]) for x in dataset ]

    plot.ylim(min(values) / 2 - 1, max(values) * 2 + 1)

    plot.scatter(labels, values, c = 'g', label='Data points')
    plot.plot(labels, perdictedValues, color='b', label='Predicted regression')

    plot.xlabel("x values")
    plot.ylabel("y values")

    plot.title("Perceptron regression")

    plot.legend()

def displayPerceptronErrors(errors):
    plot.plot([i for i in range(len(errors))], errors)

    plot.ylabel("Errors")
    plot.xlabel("Epochs")
    
    plot.title("Perceptron errors, as epochs")
    

def main():
    # func = lambda x : 2 * x + 7
    # func = lambda x : random() * 1e2 * x + random() * 1e2
    func = lambda x : x

    lim = (50, 53)

    dataset = []
    epochs = 10
    step = 0.1

    for i in (x * step for x in range(math.floor(lim[0] * 1/step), math.ceil(lim[1] * (1/step)))):
        funcValue = func(i)

        if(funcValue < 0):
            dataset.append(([func(i)], 0))
        else:
            dataset.append(([func(i)], 1))
        # points.append((i, func(i)))
    
    print("Generated dataset: ", dataset)

    perceptron = Perceptron(inputsCount = 1,
                            activation = SigmoidFunction(),
                            # error =  MSEFunction(),
                            error = CrossEntropyFunction(),
                            seed = None)

    print("Started training...")

    trainingErrors = perceptron.train(dataset, epochs)

    print("Ended training\n")

    print("Summary: ")

    for i in range(len(perceptron.weights)):
        print(f"w{i}: ", perceptron.weights[i])
    
    errors = []

    for p in dataset:
        prediction = perceptron.predict(p[0])
        singleValue = p[1]

        predictionDeviate = prediction - singleValue
        singleError = math.pow(predictionDeviate, 2)

        errors.append(singleError)  
    
    meanError = sum(errors) / len(errors)

    print("Perceptron mean error: ", meanError)

    print(trainingErrors)

    displayDatasetWithPerceptronPredictions(dataset, perceptron)
    displayPerceptronErrors(trainingErrors)

    plot.ion()
    plot.show()

    makePredictionHoldFromPerceptron(perceptron)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Recieved KeyboardInterrupt, gracefully exiting")
        sys.exit()