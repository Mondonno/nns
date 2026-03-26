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

from .playground import *

from .

# 1. czym jest siec
# 2. czym jest neuron
# 3. czym jest warstwa
# 4. czym są epoki
# 5. wpływ warstw
# 6. wpływ learning rate
# 7. wpływ funkcji
# 8. inne sieci

class StaticSineLayer(Dense):
    # inputsCount, neuronsCount, activation, seed, initializator = None, weights = None
    def __init__(self):
        super().__init__(1, 1, SineFunction(), seed=10, initializator=None, weights=[[1, 0]])
        # n n_w n_b i -> i * 

model = Sequential([
    # Dense(1, 1, LinearFunction(), seed = None),
    # Dense(1, 1, LinearFunction(), seed = None),
    # Dense(2, 4, LinearFunction(), seed = None),
    # Dense(4, 2, LinearFunction(), seed = None),
    # Dense(1, 1, SineFunction(), seed = None),
    StaticSineLayer(),
    # Dense(2, 2, RectifiedLinearFunction(), seed = None),
    # Dense(2, 2, RectifiedLinearFunction(), seed = None),
    # Dense(2, 2, RectifiedLinearFunction(), seed = None),

    # Dense(2, 2, SineFunction(), seed = None),
   Dense(1, 1, LinearFunction(), seed = 10)
], MSEFunction(), CustomLearningRateFunction(), optimizers=[
#    MomentumOptimizerFunction()
   # ClippingOptimizerFunction()
])

class CustomInputFunction(Function):
    def __init__(self):
        super().__init__()

    def call(self, input):
        # (1.1 * random.random() - 1)
        return input 
    
modelLoadAgreement = promptAgreement("Wanna load model and predict?")

modelInUse = model
trainModel = True

if modelLoadAgreement:
    modelPaths = getLastNSavedModelPaths(10, MODELS_FOLDER_PATH)
    singleModelPath = promptChoosingSinglePathFromPaths(modelPaths)

    encodedModelAsString = singleModelPath.read_text()

    encodedModel = decodeModelFromString(encodedModelAsString)
    prettyPrintOfModel = prettyPrintEncodedModel(encodedModel)

    print("Pretty print of encoded model", prettyPrintOfModel)

    importedModel = importModelWithDefaultObjects(Sequential, encodedModel)

    modelInUse = importedModel
    trainModel = False

ioSize, inputs, outputs = generateIO(CustomInputFunction(), CustomSineFunction())

print("Generated training IO size:", ioSize)
print("Inputs sample", inputs[:10])  # Display first 10 inputs for quick overview
print("Outputs sample", outputs[:10])  # Display first 10 outputs for quick overview

if trainModel:
    dataset = Dataset(ioSize, inputs, outputs, 128, seed=1)

    try:
        modelInUse.fit(dataset, 2000)
    except Exception as e:
        print("Saving model for future use, error while training occured", e)
        raise
 
    encodedModel = encodeModel(modelInUse)
    encodedModelAsString = encodedModelToString(encodedModel)

    modelPath = saveModelToFile(encodedModelAsString, MODELS_FOLDER_PATH)
    print("Path to saved model, as it is", modelPath.absolute())
else:
    print("Model training skipped, flag is set to false")

print("Coverage test for model, result", modelInUse.forwardPassByOutputLayer([
    3
]))

inputsDecoded, outputsDecoded = decodeIO(inputs, outputs)
predictedValues = predictFromInputs(modelInUse, inputs)

print("Inputs decoded:", len(inputsDecoded))  # Display first 10 inputs for quick overview
print("Outputs decoded:", len(outputsDecoded))  # Display first 10 outputs for quick overview
print("Predicted values", len(predictedValues))  # Display first 10 predicted values for quick overvi

createPlotFromIO(inputsDecoded, outputsDecoded, predictedValues)
savePlotFigureToFile(PLOT_FIGS_FOLDER)

displayPlot()
