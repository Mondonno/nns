import matplotlib.pyplot as plot

import time
import pathlib

import json

from .functions.custom.linear import CustomLinearFunction
from .functions.custom.sine import CustomSineFunction, CustomSineSimplifiedFunction
from .functions.custom.sinel import SineLinearFunction

from .functions.custom.learning.rate import CustomLearningRateFunction

from ..core.models.sequential import Sequential

from ..core.layers.dense import Dense
from ..core.layers.blank import Blank

from ..core.datasets.dataset import Dataset

from ..core.functions.function import Function

from ..core.functions.mse import MSEFunction
from ..core.functions.linear import LinearFunction
from ..core.functions.relu import RectifiedLinearFunction
from ..core.functions.sine import SineFunction

from ..core.models.optimizers import ClippingOptimizerFunction, MomentumOptimizerFunction

from ..core.dict import DictEncoder

MODELS_FOLDER = "./models"
PLOT_FIGS_FOLDER = "./figs"

MODELS_FOLDER_PATH = pathlib.Path("./models")
PLOT_FIGS_FOLDER_PATH = pathlib.Path("./figs")

MODELS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
PLOT_FIGS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

def tryParseInt(string):
    try:
        integer = int(string)
        return integer, True
    except:
        return None, False

def rangeWithFloatStep(start, stop, step = 1):
    rangeList = []

    while(start < stop):
        rangeList.append(start)

        start += step
    
    return rangeList

def getLastNSavedModelPaths(n: int, modelsPath: pathlib.Path):
    modelPaths = sorted(list(modelsPath.iterdir()), key=lambda path: path.name, reverse=True)
    nSavedModelPaths = []

    for i in range(len(modelPaths)):
       if i >= n:
           break

       singleModelPath = modelPaths[i]
       nSavedModelPaths.append(singleModelPath)
    
    return nSavedModelPaths

def promptChoosingSinglePathFromPaths(paths: list[pathlib.Path]):
    print("Choose one of following paths: ")

    for i in range(len(paths)):
        singlePath = paths[i]
        print(f"\t{i}", singlePath.name)

    print("Type one of indexes: ")

    chosenPathIndex = input()
    chosenPathIndexAsInt, hasIndexParseBeenSuccessfull = tryParseInt(chosenPathIndex)

    if not hasIndexParseBeenSuccessfull:
        print("Incorrect index provided, repeat.")
        return promptChoosingSinglePathFromPaths(paths)
    
    if chosenPathIndexAsInt >= len(paths) or chosenPathIndexAsInt < 0:
        print("Incorrect index provided, repeat.")
        return promptChoosingSinglePathFromPaths(paths)
    
    return paths[chosenPathIndexAsInt]

def promptAgreement(message: str):
    print("Agreement entered the chat")
    print(message)

    print("Y/N ", end="")

    agreementInput = input()

    agreementStrings = [
        "Y","y", "yes", "Yes"
    ]

    rejectionStrings = [
        "N", "n", "no", "No"
    ]
    
    if agreementInput in agreementStrings:
        return True
    elif agreementInput in rejectionStrings:
        return False

def generateIO(inputFunction: Function, outputFunction: Function):
    inputs = []
    outputs = []

    iterations = rangeWithFloatStep(0, 12, 0.02)

    for i in iterations:
        # (1.1 * random.random() - 1)
        inputs.append([ inputFunction.call(i) ])
        outputs.append([ outputFunction.call(i) ])

    return len(iterations), inputs, outputs

def decodeIO(inputs, outputs):
    inputsDecoded = [ x[0] for x in inputs ]
    outputsDecoded = [ x[0] for x in outputs ]

    return inputsDecoded, outputsDecoded

def predictFromInputs(model, inputs):
    predictedValues = [ model.forwardPassByOutputLayer(i)[0][0] for i in inputs ]

    return predictedValues

def createPlotFromIO(inputsDecoded, outputsDecoded, predictedValues):
    plot.ylim(min(outputsDecoded) / 2 - 1, max(outputsDecoded) * 2 + 1)

    plot.scatter(inputsDecoded, outputsDecoded, c = 'g', label='Data points')
    plot.plot(inputsDecoded, predictedValues, color='b', label='Predicted regression')

    plot.xlabel("x values")
    plot.ylabel("y values")

    plot.title("Perceptron regression")

    plot.legend()

def displayPlot():
    plot.show()

def savePlotFigureToFile(figuresPlotsPath):
    plot.savefig(f"{figuresPlotsPath}/figure_{int(time.time())}.png")

def encodeModel(model):
    dictEncoder = DictEncoder()
    return dictEncoder.encodeObject(model)

def prettyPrintEncodedModel(encodedModel):
    return json.dumps(encodedModel, indent=4)

def encodedModelToString(encodedModel):
    return json.dumps(encodedModel)

def saveModelToFile(modelString, modelsPath: pathlib.Path):
    singleModelPath = modelsPath / f"model_{int(time.time())}.model"
    singleModelPath.touch(exist_ok=True)

    singleModelPath.write_text(modelString)

    return singleModelPath

def decodeModelFromString(encodedModelAsString):
    return json.loads(encodedModelAsString)

def importModelWithDefaultObjects(modelType, encodedModel):
    defaultDict = DictEncoder().encodeTypes([
        CustomLearningRateFunction,
        SineLinearFunction,
        CustomSineFunction,
        CustomSineSimplifiedFunction,
        CustomLinearFunction
    ])

    return modelType.fromDict(encodedModel, defaultDict)

def calculateAccuracy(predictedValues, outputs, tolerance=1e-6):
    correctPredictions = 0
    totalPredictions = len(predictedValues)

    for i in range(totalPredictions):
        # Use a tolerance for floating point comparison
        if all(abs(predictedValues[i][j] - outputs[i][j]) < tolerance for j in range(len(outputs[i]))):
            correctPredictions += 1

    return correctPredictions / totalPredictions if totalPredictions > 0 else 0