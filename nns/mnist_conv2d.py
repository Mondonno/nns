import json
from pathlib import Path

import matplotlib.pyplot as plot
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from nns.core.datasets.dataset import Dataset
from nns.core.datasets.mnist_dataset import MnistDataset
from nns.core.functions.crossentropy import CrossEntropyFunction
from nns.core.functions.function import Function
from nns.core.functions.relu import RectifiedLinearFunction
from nns.core.functions.softmax import SoftmaxFunction
from nns.core.layers.activation import ActivationLayer
from nns.core.layers.convolution2d import Convolution2D
from nns.core.layers.dense import Dense
from nns.core.layers.flatten import Flatten
from nns.core.layers.maxpool2d import MaxPooling2D
from nns.core.models.sequential import Sequential

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
PAPER_FIGURES_DIRECTORY = PROJECT_ROOT / "article" / "figures"
PNG_DATASET_DIRECTORY = WORKSPACE_ROOT / "mnist-pngs-main"
IDX_DATASET_DIRECTORY = WORKSPACE_ROOT / "mnist"

TRAIN_SAMPLES_PER_CLASS = 20
TEST_SAMPLES_PER_CLASS = 8
BATCH_SIZE = 20
EPOCHS = 8

CONV1_CHANNELS = 4
CONV2_CHANNELS = 4
DENSE_INPUTS = CONV2_CHANNELS * 12 * 12


class ExperimentLearningRateFunction(Function):
    def call(self, epochIndex):
        if epochIndex < 2:
            return 0.01

        if epochIndex < 5:
            return 0.005

        return 0.002


def one_hot_encode(label, classesCount=10):
    encodedLabel = [0 for _ in range(classesCount)]
    encodedLabel[label] = 1
    return encodedLabel


def load_png_image(filePath):
    with Image.open(filePath) as image:
        grayscaleImage = image.convert("L")
        if hasattr(grayscaleImage, "get_flattened_data"):
            pixels = list(grayscaleImage.get_flattened_data())
        else:
            pixels = list(grayscaleImage.getdata())

    return [
        [pixels[rowIndex * 28 + columnIndex] / 255 for columnIndex in range(28)]
        for rowIndex in range(28)
    ]


def load_png_split(splitName, samplesPerClass):
    splitDirectory = PNG_DATASET_DIRECTORY / splitName
    inputs = []
    outputs = []
    labels = []

    for labelDirectory in sorted(splitDirectory.iterdir(), key=lambda path: path.name):
        if not labelDirectory.is_dir() or labelDirectory.name.startswith("."):
            continue

        label = int(labelDirectory.name)
        imagePaths = sorted(labelDirectory.glob("*.png"))[:samplesPerClass]

        for imagePath in imagePaths:
            inputs.append(load_png_image(imagePath))
            outputs.append(one_hot_encode(label))
            labels.append(label)

    return inputs, outputs, labels


def load_idx_split(imagesFilePath, labelsFilePath, limit):
    dataset = MnistDataset(
        imagesFilePath=imagesFilePath,
        labelsFilePath=labelsFilePath,
        batchSize=BATCH_SIZE,
        limit=limit,
        normalize=True,
        oneHot=True,
        lazy=True,
        seed=0,
    )

    inputs = [dataset.loadImage(index) for index in range(dataset.size)]
    outputs = dataset.outputs[:dataset.size]
    labels = dataset.labels[:dataset.size]
    return inputs, outputs, labels


def load_mnist_data():
    trainImagesFilePath = IDX_DATASET_DIRECTORY / "train-images-idx3-ubyte"
    trainLabelsFilePath = IDX_DATASET_DIRECTORY / "train-labels-idx1-ubyte"
    testImagesFilePath = IDX_DATASET_DIRECTORY / "t10k-images-idx3-ubyte"
    testLabelsFilePath = IDX_DATASET_DIRECTORY / "t10k-labels-idx1-ubyte"

    if all(path.exists() for path in [
        trainImagesFilePath,
        trainLabelsFilePath,
        testImagesFilePath,
        testLabelsFilePath,
    ]):
        trainLimit = TRAIN_SAMPLES_PER_CLASS * 10
        testLimit = TEST_SAMPLES_PER_CLASS * 10
        trainInputs, trainOutputs, trainLabels = load_idx_split(trainImagesFilePath, trainLabelsFilePath, trainLimit)
        testInputs, testOutputs, testLabels = load_idx_split(testImagesFilePath, testLabelsFilePath, testLimit)
        return "idx", trainInputs, trainOutputs, trainLabels, testInputs, testOutputs, testLabels

    trainInputs, trainOutputs, trainLabels = load_png_split("train", TRAIN_SAMPLES_PER_CLASS)
    testInputs, testOutputs, testLabels = load_png_split("test", TEST_SAMPLES_PER_CLASS)
    return "png", trainInputs, trainOutputs, trainLabels, testInputs, testOutputs, testLabels


def create_model():
    return Sequential([
        Convolution2D(out_channels=CONV1_CHANNELS, kernel_size=(3, 3), seed=1),
        ActivationLayer(RectifiedLinearFunction()),
        Convolution2D(out_channels=CONV2_CHANNELS, kernel_size=(3, 3), seed=2),
        ActivationLayer(RectifiedLinearFunction()),
        MaxPooling2D(poolSize=(2, 2)),
        Flatten(),
        Dense(DENSE_INPUTS, 10, SoftmaxFunction(), seed=3),
    ], CrossEntropyFunction(), ExperimentLearningRateFunction())


def unwrap_output(output):
    return [value[0] if isinstance(value, list) else value for value in output]


def predict_label(model, inputData):
    rawOutput = model.forwardPassByOutputLayer(inputData)
    scores = unwrap_output(rawOutput)
    predictedLabel = max(range(len(scores)), key=lambda index: scores[index])
    return predictedLabel, scores


def calculate_mean_loss(model, inputs, outputs):
    losses = []

    for inputData, expectedOutput in zip(inputs, outputs):
        _, scores = predict_label(model, inputData)
        sampleLoss = sum(
            model.error.call((score, expectedValue))
            for score, expectedValue in zip(scores, expectedOutput)
        )
        losses.append(sampleLoss)

    return sum(losses) / len(losses)


def calculate_accuracy(model, inputs, labels):
    correctPredictions = 0

    for inputData, label in zip(inputs, labels):
        prediction, _ = predict_label(model, inputData)
        if prediction == label:
            correctPredictions += 1

    return correctPredictions / len(labels)


def build_confusion_matrix(model, inputs, labels):
    confusionMatrix = [[0 for _ in range(10)] for _ in range(10)]

    for inputData, label in zip(inputs, labels):
        prediction, _ = predict_label(model, inputData)
        confusionMatrix[label][prediction] += 1

    return confusionMatrix


def save_training_curves(history, outputPath):
    epochs = [entry["epoch"] for entry in history]
    losses = [entry["train_loss"] for entry in history]
    trainAccuracies = [entry["train_accuracy"] for entry in history]
    testAccuracies = [entry["test_accuracy"] for entry in history]

    figure, axes = plot.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, losses, color="#c75b12", linewidth=2.2, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, trainAccuracies, color="#6d904f", linewidth=2.2, marker="o", label="Train")
    axes[1].plot(epochs, testAccuracies, color="#1f6f8b", linewidth=2.2, marker="o", label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(outputPath, dpi=200, bbox_inches="tight")
    plot.close(figure)


def save_dataset_samples(images, labels, outputPath):
    figure, axes = plot.subplots(2, 5, figsize=(10, 4.3))

    for digit in range(10):
        imageIndex = labels.index(digit)
        axis = axes[digit // 5][digit % 5]
        axis.imshow(images[imageIndex], cmap="gray")
        axis.set_title(f"Digit {digit}")
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(outputPath, dpi=200, bbox_inches="tight")
    plot.close(figure)


def save_confusion_matrix(confusionMatrix, outputPath):
    figure, axis = plot.subplots(figsize=(6.3, 5.4))
    heatmap = axis.imshow(confusionMatrix, cmap="Blues")
    axis.set_title("MNIST Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))

    for rowIndex in range(10):
        for columnIndex in range(10):
            axis.text(columnIndex, rowIndex, str(confusionMatrix[rowIndex][columnIndex]), ha="center", va="center", color="#132238", fontsize=8)

    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(outputPath, dpi=200, bbox_inches="tight")
    plot.close(figure)


def shape_of(values):
    if isinstance(values, list):
        if len(values) == 0:
            return [0]

        return [len(values)] + shape_of(values[0])

    return []


def format_shape(shape):
    filteredShape = list(shape)

    if len(filteredShape) == 2 and filteredShape[-1] == 1:
        filteredShape = filteredShape[:-1]

    return " x ".join(str(dimension) for dimension in filteredShape)


def layer_label(layer):
    if isinstance(layer, Convolution2D):
        return "Conv2D"

    if isinstance(layer, ActivationLayer):
        if isinstance(layer.activation, RectifiedLinearFunction):
            return "ReLU"

        return layer.activation.__class__.__name__.replace("Function", "")

    if isinstance(layer, MaxPooling2D):
        return "MaxPool"

    if isinstance(layer, Flatten):
        return "Flatten"

    if isinstance(layer, Dense):
        if isinstance(layer.activation, SoftmaxFunction):
            return "Dense + Softmax"

        return "Dense"

    return layer.__class__.__name__


def layer_color(layer):
    if isinstance(layer, Convolution2D):
        return "#fce5cd"

    if isinstance(layer, ActivationLayer):
        return "#f4cccc"

    if isinstance(layer, MaxPooling2D):
        return "#cfe2f3"

    if isinstance(layer, Flatten):
        return "#d9d2e9"

    if isinstance(layer, Dense):
        return "#ead1dc"

    return "#d9ead3"


def trace_model(model, sampleInput):
    currentOutput = [sampleInput]
    tracedBlocks = [{
        "name": "Input",
        "shape": shape_of(currentOutput),
        "color": "#d9ead3",
    }]

    for layer in model.layers:
        forwardResult = layer.forwardPass(currentOutput)

        if isinstance(forwardResult, tuple):
            _, currentOutput = forwardResult
        else:
            currentOutput = forwardResult

        tracedBlocks.append({
            "name": layer_label(layer),
            "shape": shape_of(currentOutput),
            "color": layer_color(layer),
        })

    return tracedBlocks


def save_architecture_diagram(model, sampleInput, outputPath):
    blocks = trace_model(model, sampleInput)
    figureWidth = max(12, len(blocks) * 1.7)
    figure, axis = plot.subplots(figsize=(figureWidth, 3.1))

    axis.set_xlim(0, len(blocks) * 1.9 + 0.8)
    axis.set_ylim(0, 2.4)
    axis.axis("off")

    xPosition = 0.6

    for index, block in enumerate(blocks):
        box = FancyBboxPatch(
            (xPosition, 0.75),
            1.42,
            0.82,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#1f1f1f",
            facecolor=block["color"],
        )
        axis.add_patch(box)
        axis.text(
            xPosition + 0.71,
            1.16,
            f"{block['name']}\n{format_shape(block['shape'])}",
            ha="center",
            va="center",
            fontsize=10,
        )

        if index < len(blocks) - 1:
            arrow = FancyArrowPatch(
                (xPosition + 1.45, 1.16),
                (xPosition + 2.08, 1.16),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.4,
                color="#1f1f1f",
            )
            axis.add_patch(arrow)

        xPosition += 1.9

    figure.tight_layout()
    figure.savefig(outputPath, dpi=200, bbox_inches="tight")
    plot.close(figure)


def train_and_evaluate():
    PAPER_FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    datasetSource, trainInputs, trainOutputs, trainLabels, testInputs, testOutputs, testLabels = load_mnist_data()

    trainDataset = Dataset(
        size=len(trainInputs),
        inputs=trainInputs,
        outputs=trainOutputs,
        batchSize=BATCH_SIZE,
        seed=0,
        appendBatchResidue=False,
    )

    model = create_model()
    history = []

    for epoch in range(1, EPOCHS + 1):
        list(model.fitWithGradientTape(trainDataset, 1))

        trainLoss = calculate_mean_loss(model, trainInputs, trainOutputs)
        trainAccuracy = calculate_accuracy(model, trainInputs, trainLabels)
        testAccuracy = calculate_accuracy(model, testInputs, testLabels)

        history.append({
            "epoch": epoch,
            "train_loss": trainLoss,
            "train_accuracy": trainAccuracy,
            "test_accuracy": testAccuracy,
        })

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={trainLoss:.4f} | "
            f"train_accuracy={trainAccuracy:.4f} | "
            f"test_accuracy={testAccuracy:.4f}",
            flush=True,
        )

    confusionMatrix = build_confusion_matrix(model, testInputs, testLabels)

    save_training_curves(history, PAPER_FIGURES_DIRECTORY / "mnist_training_curves.png")
    save_dataset_samples(testInputs, testLabels, PAPER_FIGURES_DIRECTORY / "mnist_dataset_samples.png")
    save_confusion_matrix(confusionMatrix, PAPER_FIGURES_DIRECTORY / "mnist_confusion_matrix.png")
    save_architecture_diagram(model, testInputs[0], PAPER_FIGURES_DIRECTORY / "mnist_network_architecture.png")

    metrics = {
        "dataset_source": datasetSource,
        "train_samples": len(trainInputs),
        "test_samples": len(testInputs),
        "epochs": EPOCHS,
        "history": history,
        "architecture": [layer_label(layer) for layer in model.layers],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
    }

    metricsPath = PAPER_FIGURES_DIRECTORY / "mnist_conv2d_metrics.json"
    metricsPath.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_evaluate()
