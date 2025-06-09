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
from ..core.layers.convolution2d import Convolution2D
from ..core.layers.maxpool2d import MaxPooling2D
from ..core.layers.flatten import Flatten
from ..core.layers.blank import Blank

from ..core.datasets.dataset import Dataset

from ..core.functions.function import Function

from ..core.functions.mse import MSEFunction
from ..core.functions.linear import LinearFunction
from ..core.functions.relu import RectifiedLinearFunction
from ..core.functions.sine import SineFunction

from ..core.datasets.image_dataset import ImageDataset

from ..core.datasets.image_dataset import ImageDataset
from ..core.datasets.transforms.min_max import MinMaxTransformFunction
from ..core.models.optimizers import ClippingOptimizerFunction, MomentumOptimizerFunction
from ..core.dict import DictEncoder

from .playground import *

class LabelFunction(Function):
    def __init__(self):
        super().__init__()

    def call(self, filepath):
        import os
        return int(os.path.basename(os.path.dirname(filepath)))

class TransformFunction(MinMaxTransformFunction):
    def __init__(self):
        super().__init__(min_value=0, max_value=255)

    def call(self, imageArray):
        print(imageArray)
        for i in range(len(imageArray)):
            for j in range(len(imageArray[i])):
                print(imageArray[i][j])
                imageArray[i][j] = super().call(imageArray[i][j])
        return imageArray

# get home directory
HOME_DIRECTORY = pathlib.Path.home()
MNIST_DIRECTORY = HOME_DIRECTORY / "Documents" / "Projects" / "learning" / "mnist-pngs-main" / "train"

mnist_dataset = ImageDataset(
    directory=MNIST_DIRECTORY,
    batchSize=32,                         # batch size
    transformFunction=TransformFunction(),               
    appendBatchResidue=False,             # usually False
    labelFunction=LabelFunction,    # function to extract label from filename
    imageShape=(28, 28),                  # MNIST images are 28x28
    grayscale=True                        # MNIST is grayscale
)

model = Sequential([
    MaxPooling2D(poolSize=(2,2)),
    Convolution2D(out_channels=2, kernel_size=(3, 3)),
    MaxPooling2D(poolSize=(2,2)),
    Convolution2D(out_channels=3, kernel_size=(3, 3)),
    MaxPooling2D(poolSize=(2,2)),
    Convolution2D(out_channels=3, kernel_size=(3, 3)),
    Flatten(),
    Dense(12, 10, RectifiedLinearFunction(), seed=42),
], MSEFunction(), CustomLearningRateFunction(), optimizers=[
    # MomentumOptimizerFunction()
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

ioSize = mnist_dataset.size
inputs = mnist_dataset.inputs
outputs = mnist_dataset.outputs

print("Gathered training IO size:", ioSize)

if trainModel:
    try:
        modelInUse.fit(mnist_dataset, 100)
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

predictedValues = predictFromInputs(modelInUse, inputs)

print("Inputs:", len(inputs))  
print("Outputs:", len(outputs)) 
print("Predicted:", len(predictedValues)) 
print("Accuracy", calculateAccuracy(outputs, predictedValues))

# createPlotFromIO(inputsDecoded, outputsDecoded, predictedValues)
savePlotFigureToFile(PLOT_FIGS_FOLDER)

displayPlot()
