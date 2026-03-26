import json
from pathlib import Path

import matplotlib.pyplot as plot
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from nns.core.datasets.dataset import Dataset
from nns.core.datasets.mnist_dataset import MnistDataset
from nns.core.functions.function import Function
from nns.core.functions.linear import LinearFunction
from nns.core.functions.mse import MSEFunction
from nns.core.layers.convolution2d import Convolution2D
from nns.core.layers.dense import Dense
from nns.core.layers.flatten import Flatten
from nns.core.layers.maxpool2d import MaxPooling2D
from nns.core.models.sequential import Sequential

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES_DIRECTORY = PROJECT_ROOT / "mnist-in-pure-python" / "figures"
PNG_DATASET_DIRECTORY = PROJECT_ROOT / "mnist-pngs-main"
IDX_DATASET_DIRECTORY = PROJECT_ROOT / "mnist"

TRAIN_SAMPLES_PER_CLASS = 8
TEST_SAMPLES_PER_CLASS = 4
BATCH_SIZE = 10
EPOCHS = 4


class ExperimentLearningRateFunction(Function):
    def call(self, epochIndex):
        return 0.0001


def one_hot_encode(label, classesCount=10):
    encodedLabel = [0 for _ in range(classesCount)]
    encodedLabel[label] = 1
    return encodedLabel


def load_png_image(filePath):
    with Image.open(filePath) as image:
        grayscaleImage = image.convert("L")
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
        Convolution2D(out_channels=2, kernel_size=(3, 3), seed=1),
        Convolution2D(out_channels=2, kernel_size=(3, 3), seed=2),
        MaxPooling2D(poolSize=(2, 2)),
        Flatten(),
        Dense(288, 10, LinearFunction(), seed=3),
    ], MSEFunction(), ExperimentLearningRateFunction())


def unwrap_output(output):
    return [value[0] if isinstance(value, list) else value for value in output]


def predict_label(model, inputData):
    rawOutput = model.forwardPassByOutputLayer(inputData)
    scores = unwrap_output(rawOutput)
    predictedLabel = max(range(len(scores)), key=lambda index: scores[index])
    return predictedLabel, scores


def calculate_mean_loss(model, inputs, outputs):
    mse = MSEFunction()
    losses = []

    for inputData, expectedOutput in zip(inputs, outputs):
        _, scores = predict_label(model, inputData)
        sampleLoss = 0

        for score, expectedValue in zip(scores, expectedOutput):
            sampleLoss += mse.call((score, expectedValue))

        losses.append(sampleLoss / len(expectedOutput))

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
    accuracies = [entry["test_accuracy"] for entry in history]

    figure, axes = plot.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, losses, color="#c75b12", linewidth=2.2, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, accuracies, color="#1f6f8b", linewidth=2.2, marker="o")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)

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


def save_architecture_diagram(outputPath):
    blocks = [
        ("Input\n1 x 28 x 28", "#d9ead3"),
        ("Conv2D\n4 x 26 x 26", "#fce5cd"),
        ("Conv2D\n4 x 24 x 24", "#f9cb9c"),
        ("MaxPool\n4 x 12 x 12", "#cfe2f3"),
        ("Flatten\n576", "#d9d2e9"),
        ("Dense\n10", "#ead1dc"),
    ]

    figure, axis = plot.subplots(figsize=(12, 2.8))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 2.2)
    axis.axis("off")

    xPosition = 0.6
    for index, (label, color) in enumerate(blocks):
        block = FancyBboxPatch(
            (xPosition, 0.75),
            1.4,
            0.7,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#1f1f1f",
            facecolor=color,
        )
        axis.add_patch(block)
        axis.text(xPosition + 0.7, 1.1, label, ha="center", va="center", fontsize=10)

        if index < len(blocks) - 1:
            arrow = FancyArrowPatch((xPosition + 1.42, 1.1), (xPosition + 2.1, 1.1), arrowstyle="->", mutation_scale=14, linewidth=1.4, color="#1f1f1f")
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
            f"test_accuracy={testAccuracy:.4f}"
        , flush=True)

    confusionMatrix = build_confusion_matrix(model, testInputs, testLabels)

    save_training_curves(history, PAPER_FIGURES_DIRECTORY / "mnist_training_curves.png")
    save_dataset_samples(testInputs, testLabels, PAPER_FIGURES_DIRECTORY / "mnist_dataset_samples.png")
    save_confusion_matrix(confusionMatrix, PAPER_FIGURES_DIRECTORY / "mnist_confusion_matrix.png")
    save_architecture_diagram(PAPER_FIGURES_DIRECTORY / "mnist_network_architecture.png")

    metrics = {
        "dataset_source": datasetSource,
        "train_samples": len(trainInputs),
        "test_samples": len(testInputs),
        "epochs": EPOCHS,
        "history": history,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
    }

    metricsPath = PAPER_FIGURES_DIRECTORY / "mnist_conv2d_metrics.json"
    metricsPath.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_evaluate()
