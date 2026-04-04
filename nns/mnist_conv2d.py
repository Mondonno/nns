import json
from pathlib import Path

import matplotlib.pyplot as plot
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

SUBSET_TRAIN_SAMPLES_PER_CLASS = 20
SUBSET_TEST_SAMPLES_PER_CLASS = 8
SUBSET_BATCH_SIZE = 20
SUBSET_EPOCHS = 8

FULL_BATCH_SIZE = 100
FULL_EPOCHS = 8
FULL_TRAIN_EVAL_LIMIT = 2000
FULL_TEST_EVAL_LIMIT = 10000

CONV1_CHANNELS = 4
CONV2_CHANNELS = 4
SUBSET_DENSE_INPUTS = CONV2_CHANNELS * 12 * 12


class SubsetLearningRateFunction(Function):
    def call(self, epochIndex):
        if epochIndex < 2:
            return 0.01

        if epochIndex < 5:
            return 0.005

        return 0.002


class FullLearningRateFunction(Function):
    def call(self, epochIndex):
        if epochIndex < 1:
            return 0.03

        return 0.01


def find_existing_path(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise ValueError(f"None of the candidate paths exists: {candidates}")


def resolve_idx_paths():
    trainImagesFilePath = find_existing_path([
        WORKSPACE_ROOT / "mnist" / "train-images-idx3-ubyte",
        WORKSPACE_ROOT / "mnist" / "train-images.idx3-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "train-images-idx3-ubyte" / "train-images-idx3-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "train-images.idx3-ubyte",
    ])
    trainLabelsFilePath = find_existing_path([
        WORKSPACE_ROOT / "mnist" / "train-labels-idx1-ubyte",
        WORKSPACE_ROOT / "mnist" / "train-labels.idx1-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "train-labels.idx1-ubyte",
    ])
    testImagesFilePath = find_existing_path([
        WORKSPACE_ROOT / "mnist" / "t10k-images-idx3-ubyte",
        WORKSPACE_ROOT / "mnist" / "t10k-images.idx3-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "t10k-images.idx3-ubyte",
    ])
    testLabelsFilePath = find_existing_path([
        WORKSPACE_ROOT / "mnist" / "t10k-labels-idx1-ubyte",
        WORKSPACE_ROOT / "mnist" / "t10k-labels.idx1-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte",
        WORKSPACE_ROOT / "mnist-archive" / "t10k-labels.idx1-ubyte",
    ])

    return trainImagesFilePath, trainLabelsFilePath, testImagesFilePath, testLabelsFilePath


def one_hot_encode(label, classesCount=10):
    encodedLabel = [0 for _ in range(classesCount)]
    encodedLabel[label] = 1
    return encodedLabel


def load_balanced_idx_split(imagesFilePath, labelsFilePath, samplesPerClass):
    dataset = MnistDataset(
        imagesFilePath=imagesFilePath,
        labelsFilePath=labelsFilePath,
        batchSize=max(1, samplesPerClass),
        normalize=True,
        oneHot=True,
        lazy=True,
        seed=0,
    )

    selectedInputs = []
    selectedOutputs = []
    selectedLabels = []
    selectedCounts = {digit: 0 for digit in range(10)}

    for index, label in enumerate(dataset.labels):
        if selectedCounts[label] >= samplesPerClass:
            continue

        selectedInputs.append(dataset.loadImage(index))
        selectedOutputs.append(dataset.outputs[index])
        selectedLabels.append(label)
        selectedCounts[label] += 1

        if all(count >= samplesPerClass for count in selectedCounts.values()):
            break

    return selectedInputs, selectedOutputs, selectedLabels


def create_subset_model():
    return Sequential([
        Convolution2D(out_channels=CONV1_CHANNELS, kernel_size=(3, 3), seed=1),
        ActivationLayer(RectifiedLinearFunction()),
        Convolution2D(out_channels=CONV2_CHANNELS, kernel_size=(3, 3), seed=2),
        ActivationLayer(RectifiedLinearFunction()),
        MaxPooling2D(poolSize=(2, 2)),
        Flatten(),
        Dense(SUBSET_DENSE_INPUTS, 10, SoftmaxFunction(), seed=3),
    ], CrossEntropyFunction(), SubsetLearningRateFunction())


def create_full_model():
    # return Sequential([
    #     Flatten(),
    #     Dense(784, 10, SoftmaxFunction(), seed=11),
    # ], CrossEntropyFunction(), FullLearningRateFunction())
    return create_subset_model()


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


def calculate_dataset_accuracy(model, dataset, limit=None):
    size = dataset.size if limit is None else min(limit, dataset.size)
    correctPredictions = 0

    for index in range(size):
        prediction, _ = predict_label(model, dataset.loadImage(index))
        if prediction == dataset.labels[index]:
            correctPredictions += 1

    return correctPredictions / size


def calculate_dataset_mean_loss(model, dataset, limit):
    size = min(limit, dataset.size)
    losses = []

    for index in range(size):
        _, scores = predict_label(model, dataset.loadImage(index))
        expectedOutput = dataset.outputs[index]
        sampleLoss = sum(
            model.error.call((score, expectedValue))
            for score, expectedValue in zip(scores, expectedOutput)
        )
        losses.append(sampleLoss)

    return sum(losses) / len(losses)


def build_confusion_matrix(model, inputs, labels):
    confusionMatrix = [[0 for _ in range(10)] for _ in range(10)]

    for inputData, label in zip(inputs, labels):
        prediction, _ = predict_label(model, inputData)
        confusionMatrix[label][prediction] += 1

    return confusionMatrix


def build_dataset_confusion_matrix(model, dataset, limit=None):
    size = dataset.size if limit is None else min(limit, dataset.size)
    confusionMatrix = [[0 for _ in range(10)] for _ in range(10)]

    for index in range(size):
        label = dataset.labels[index]
        prediction, _ = predict_label(model, dataset.loadImage(index))
        confusionMatrix[label][prediction] += 1

    return confusionMatrix


def save_training_curves(subsetHistory, fullHistory, outputPath):
    figure, axes = plot.subplots(2, 2, figsize=(12, 8))

    subsetEpochs = [entry["epoch"] for entry in subsetHistory]
    subsetLosses = [entry["train_loss"] for entry in subsetHistory]
    subsetTrainAccuracies = [entry["train_accuracy"] for entry in subsetHistory]
    subsetTestAccuracies = [entry["test_accuracy"] for entry in subsetHistory]

    fullEpochs = [entry["epoch"] for entry in fullHistory]
    fullLosses = [entry["train_loss"] for entry in fullHistory]
    fullTrainAccuracies = [entry["train_accuracy"] for entry in fullHistory]
    fullTestAccuracies = [entry["test_accuracy"] for entry in fullHistory]

    axes[0][0].plot(subsetEpochs, subsetLosses, color="#c75b12", linewidth=2.2, marker="o")
    axes[0][0].set_title("Subset CNN Loss")
    axes[0][0].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Cross-Entropy")
    axes[0][0].grid(alpha=0.3)

    axes[0][1].plot(subsetEpochs, subsetTrainAccuracies, color="#6d904f", linewidth=2.2, marker="o", label="Train")
    axes[0][1].plot(subsetEpochs, subsetTestAccuracies, color="#1f6f8b", linewidth=2.2, marker="o", label="Test")
    axes[0][1].set_title("Subset CNN Accuracy")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].set_ylabel("Accuracy")
    axes[0][1].set_ylim(0, 1)
    axes[0][1].grid(alpha=0.3)
    axes[0][1].legend()

    axes[1][0].plot(fullEpochs, fullLosses, color="#a61c3c", linewidth=2.2, marker="o")
    axes[1][0].set_title("Full IDX Softmax Loss")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].set_ylabel("Cross-Entropy")
    axes[1][0].grid(alpha=0.3)

    axes[1][1].plot(fullEpochs, fullTrainAccuracies, color="#3d7d3a", linewidth=2.2, marker="o", label="Train sample")
    axes[1][1].plot(fullEpochs, fullTestAccuracies, color="#124c7c", linewidth=2.2, marker="o", label="Test")
    axes[1][1].set_title("Full IDX Softmax Accuracy")
    axes[1][1].set_xlabel("Epoch")
    axes[1][1].set_ylabel("Accuracy")
    axes[1][1].set_ylim(0, 1)
    axes[1][1].grid(alpha=0.3)
    axes[1][1].legend()

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


def save_confusion_matrices(subsetMatrix, fullMatrix, outputPath):
    figure, axes = plot.subplots(1, 2, figsize=(12, 5.2))

    for axis, matrix, title in [
        (axes[0], subsetMatrix, "Subset CNN"),
        (axes[1], fullMatrix, "Full IDX Softmax"),
    ]:
        heatmap = axis.imshow(matrix, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))

        for rowIndex in range(10):
            for columnIndex in range(10):
                axis.text(columnIndex, rowIndex, str(matrix[rowIndex][columnIndex]), ha="center", va="center", color="#132238", fontsize=7)

        figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    figure.savefig(outputPath, dpi=200, bbox_inches="tight")
    plot.close(figure)


def save_experiment_comparison(subsetMetrics, fullMetrics, outputPath):
    figure, axes = plot.subplots(1, 2, figsize=(11.5, 4.2))

    experimentNames = ["Subset CNN", "Full IDX Softmax"]
    testAccuracies = [subsetMetrics["final_test_accuracy"], fullMetrics["final_test_accuracy"]]
    trainSampleCounts = [subsetMetrics["train_samples"], fullMetrics["train_samples"]]
    testSampleCounts = [subsetMetrics["test_samples"], fullMetrics["test_samples"]]

    axes[0].bar(experimentNames, testAccuracies, color=["#f4a261", "#2a9d8f"])
    axes[0].set_title("Final Test Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(experimentNames, trainSampleCounts, color="#6c8ebf", label="Train samples")
    axes[1].bar(experimentNames, testSampleCounts, color="#93c47d", label="Test samples")
    axes[1].set_title("Dataset Sizes")
    axes[1].set_ylabel("Samples")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

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


def draw_architecture(axis, blocks, title):
    axis.set_xlim(0, len(blocks) * 1.9 + 0.8)
    axis.set_ylim(0, 2.4)
    axis.axis("off")
    axis.set_title(title, fontsize=12)

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
            fontsize=9,
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


def save_architecture_diagram(subsetModel, subsetSampleInput, fullModel, fullSampleInput, outputPath):
    subsetBlocks = trace_model(subsetModel, subsetSampleInput)
    fullBlocks = trace_model(fullModel, fullSampleInput)

    figure, axes = plot.subplots(2, 1, figsize=(12, 5.8))
    draw_architecture(axes[0], subsetBlocks, "Subset CNN")
    draw_architecture(axes[1], fullBlocks, "Full IDX Softmax")
    figure.tight_layout()
    figure.savefig(outputPath, dpi=200, bbox_inches="tight")
    plot.close(figure)


def run_subset_experiment(trainImagesFilePath, trainLabelsFilePath, testImagesFilePath, testLabelsFilePath):
    trainInputs, trainOutputs, trainLabels = load_balanced_idx_split(
        trainImagesFilePath,
        trainLabelsFilePath,
        SUBSET_TRAIN_SAMPLES_PER_CLASS,
    )
    testInputs, testOutputs, testLabels = load_balanced_idx_split(
        testImagesFilePath,
        testLabelsFilePath,
        SUBSET_TEST_SAMPLES_PER_CLASS,
    )

    trainDataset = Dataset(
        size=len(trainInputs),
        inputs=trainInputs,
        outputs=trainOutputs,
        batchSize=SUBSET_BATCH_SIZE,
        seed=0,
        appendBatchResidue=False,
    )

    model = create_subset_model()
    history = []

    for epoch in range(1, SUBSET_EPOCHS + 1):
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
            f"[subset] epoch {epoch}/{SUBSET_EPOCHS} | "
            f"train_loss={trainLoss:.4f} | "
            f"train_accuracy={trainAccuracy:.4f} | "
            f"test_accuracy={testAccuracy:.4f}",
            flush=True,
        )

    confusionMatrix = build_confusion_matrix(model, testInputs, testLabels)

    metrics = {
        "experiment": "subset_cnn",
        "dataset_source": "idx",
        "train_samples": len(trainInputs),
        "test_samples": len(testInputs),
        "epochs": SUBSET_EPOCHS,
        "history": history,
        "architecture": [layer_label(layer) for layer in model.layers],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
    }

    return {
        "model": model,
        "train_inputs": trainInputs,
        "train_outputs": trainOutputs,
        "train_labels": trainLabels,
        "test_inputs": testInputs,
        "test_outputs": testOutputs,
        "test_labels": testLabels,
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
    }


def run_full_experiment(trainImagesFilePath, trainLabelsFilePath, testImagesFilePath, testLabelsFilePath):
    trainDataset = MnistDataset(
        imagesFilePath=trainImagesFilePath,
        labelsFilePath=trainLabelsFilePath,
        batchSize=FULL_BATCH_SIZE,
        normalize=True,
        oneHot=True,
        lazy=True,
        seed=0,
    )
    testDataset = MnistDataset(
        imagesFilePath=testImagesFilePath,
        labelsFilePath=testLabelsFilePath,
        batchSize=FULL_BATCH_SIZE,
        normalize=True,
        oneHot=True,
        lazy=True,
        seed=0,
    )

    model = create_full_model()
    history = []

    for epoch in range(1, FULL_EPOCHS + 1):
        list(model.fitWithGradientTape(trainDataset, 1))

        trainLoss = calculate_dataset_mean_loss(model, trainDataset, FULL_TRAIN_EVAL_LIMIT)
        trainAccuracy = calculate_dataset_accuracy(model, trainDataset, FULL_TRAIN_EVAL_LIMIT)
        testAccuracy = calculate_dataset_accuracy(model, testDataset, FULL_TEST_EVAL_LIMIT)

        history.append({
            "epoch": epoch,
            "train_loss": trainLoss,
            "train_accuracy": trainAccuracy,
            "test_accuracy": testAccuracy,
        })

        print(
            f"[full] epoch {epoch}/{FULL_EPOCHS} | "
            f"train_loss={trainLoss:.4f} | "
            f"train_accuracy(sample)={trainAccuracy:.4f} | "
            f"test_accuracy={testAccuracy:.4f}",
            flush=True,
        )

    confusionMatrix = build_dataset_confusion_matrix(model, testDataset, FULL_TEST_EVAL_LIMIT)

    metrics = {
        "experiment": "full_softmax",
        "dataset_source": "idx",
        "train_samples": trainDataset.size,
        "test_samples": min(FULL_TEST_EVAL_LIMIT, testDataset.size),
        "epochs": FULL_EPOCHS,
        "history": history,
        "architecture": [layer_label(layer) for layer in model.layers],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "train_accuracy_note": f"Measured on the first {FULL_TRAIN_EVAL_LIMIT} training samples after each epoch.",
    }

    return {
        "model": model,
        "train_dataset": trainDataset,
        "test_dataset": testDataset,
        "confusion_matrix": confusionMatrix,
        "metrics": metrics,
    }


def train_and_evaluate():
    PAPER_FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

    trainImagesFilePath, trainLabelsFilePath, testImagesFilePath, testLabelsFilePath = resolve_idx_paths()

    subsetExperiment = run_subset_experiment(
        trainImagesFilePath,
        trainLabelsFilePath,
        testImagesFilePath,
        testLabelsFilePath,
    )
    fullExperiment = run_full_experiment(
        trainImagesFilePath,
        trainLabelsFilePath,
        testImagesFilePath,
        testLabelsFilePath,
    )

    save_dataset_samples(
        subsetExperiment["test_inputs"],
        subsetExperiment["test_labels"],
        PAPER_FIGURES_DIRECTORY / "mnist_dataset_samples.png",
    )
    save_architecture_diagram(
        subsetExperiment["model"],
        subsetExperiment["test_inputs"][0],
        fullExperiment["model"],
        fullExperiment["test_dataset"].loadImage(0),
        PAPER_FIGURES_DIRECTORY / "mnist_network_architecture.png",
    )
    save_training_curves(
        subsetExperiment["metrics"]["history"],
        fullExperiment["metrics"]["history"],
        PAPER_FIGURES_DIRECTORY / "mnist_training_curves.png",
    )
    save_confusion_matrices(
        subsetExperiment["confusion_matrix"],
        fullExperiment["confusion_matrix"],
        PAPER_FIGURES_DIRECTORY / "mnist_confusion_matrix.png",
    )
    save_experiment_comparison(
        subsetExperiment["metrics"],
        fullExperiment["metrics"],
        PAPER_FIGURES_DIRECTORY / "mnist_experiment_comparison.png",
    )

    metrics = {
        "subset_experiment": subsetExperiment["metrics"],
        "full_experiment": fullExperiment["metrics"],
    }

    metricsPath = PAPER_FIGURES_DIRECTORY / "mnist_conv2d_metrics.json"
    metricsPath.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_evaluate()
