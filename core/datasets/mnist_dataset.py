import os
import struct

from .dataset import Dataset

MNIST_IMAGES_MAGIC = 2051
MNIST_LABELS_MAGIC = 2049


def _validate_file_path(filePath):
    if not os.path.isfile(filePath):
        raise ValueError(f"File '{filePath}' does not exist.")


def _one_hot_encode(label, classesCount=10):
    encodedLabel = [0 for _ in range(classesCount)]
    encodedLabel[label] = 1
    return encodedLabel


def parse_idx3_ubyte(filePath, limit=None, normalize=True):
    _validate_file_path(filePath)

    with open(filePath, "rb") as file:
        magic, imagesCount, rowsCount, columnsCount = struct.unpack(">IIII", file.read(16))

        if magic != MNIST_IMAGES_MAGIC:
            raise ValueError(f"Incorrect IDX image magic number: expected {MNIST_IMAGES_MAGIC}, got {magic}.")

        pixelsCount = rowsCount * columnsCount
        if limit is None:
            limit = imagesCount

        limit = min(limit, imagesCount)
        images = []

        for _ in range(limit):
            rawImage = file.read(pixelsCount)
            if len(rawImage) != pixelsCount:
                raise ValueError("Unexpected end of file while reading IDX image data.")

            image = []
            for rowIndex in range(rowsCount):
                row = []
                for columnIndex in range(columnsCount):
                    pixelIndex = rowIndex * columnsCount + columnIndex
                    pixelValue = rawImage[pixelIndex]

                    if normalize:
                        pixelValue /= 255

                    row.append(pixelValue)

                image.append(row)

            images.append(image)

        return images


def parse_idx1_ubyte(filePath, limit=None):
    _validate_file_path(filePath)

    with open(filePath, "rb") as file:
        magic, labelsCount = struct.unpack(">II", file.read(8))

        if magic != MNIST_LABELS_MAGIC:
            raise ValueError(f"Incorrect IDX label magic number: expected {MNIST_LABELS_MAGIC}, got {magic}.")

        if limit is None:
            limit = labelsCount

        limit = min(limit, labelsCount)
        rawLabels = file.read(limit)

        if len(rawLabels) != limit:
            raise ValueError("Unexpected end of file while reading IDX label data.")

        return [int(label) for label in rawLabels]


class MnistDataset(Dataset):
    def __init__(
        self,
        imagesFilePath,
        labelsFilePath,
        batchSize,
        limit=None,
        normalize=True,
        oneHot=True,
        appendBatchResidue=False,
        seed=None,
        lazy=False,
    ):
        self.imagesFilePath = imagesFilePath
        self.labelsFilePath = labelsFilePath
        self.normalize = normalize
        self.oneHot = oneHot
        self.lazy = lazy

        self.imagesCount = None
        self.rowsCount = None
        self.columnsCount = None
        self.imageBytesCount = None
        self._loadImagesMetadata()

        labels = parse_idx1_ubyte(self.labelsFilePath, limit=limit)
        self.size = min(self.imagesCount, len(labels))
        self.labels = labels[:self.size]
        self.outputs = self._prepareOutputs(self.labels)

        if self.lazy:
            self.inputs = [None for _ in range(self.size)]
        else:
            self.inputs = parse_idx3_ubyte(
                self.imagesFilePath,
                limit=self.size,
                normalize=self.normalize,
            )

        super().__init__(
            self.size,
            self.inputs,
            self.outputs,
            batchSize,
            seed=seed,
            appendBatchResidue=appendBatchResidue,
        )

    def _loadImagesMetadata(self):
        _validate_file_path(self.imagesFilePath)

        with open(self.imagesFilePath, "rb") as file:
            magic, imagesCount, rowsCount, columnsCount = struct.unpack(">IIII", file.read(16))

        if magic != MNIST_IMAGES_MAGIC:
            raise ValueError(f"Incorrect IDX image magic number: expected {MNIST_IMAGES_MAGIC}, got {magic}.")

        self.imagesCount = imagesCount
        self.rowsCount = rowsCount
        self.columnsCount = columnsCount
        self.imageBytesCount = self.rowsCount * self.columnsCount

    def _prepareOutputs(self, labels):
        if not self.oneHot:
            return labels

        return [_one_hot_encode(label) for label in labels]

    def loadImage(self, index):
        if index < 0 or index >= self.size:
            raise IndexError(f"Index {index} out of range for dataset of size {self.size}.")

        offset = 16 + index * self.imageBytesCount

        with open(self.imagesFilePath, "rb") as file:
            file.seek(offset)
            rawImage = file.read(self.imageBytesCount)

        if len(rawImage) != self.imageBytesCount:
            raise ValueError("Unexpected end of file while reading a lazy IDX image.")

        image = []

        for rowIndex in range(self.rowsCount):
            row = []
            for columnIndex in range(self.columnsCount):
                pixelIndex = rowIndex * self.columnsCount + columnIndex
                pixelValue = rawImage[pixelIndex]

                if self.normalize:
                    pixelValue /= 255

                row.append(pixelValue)

            image.append(row)

        return image

    def generate(self):
        if not self.lazy:
            return super().generate()

        dataIndicies = [i for i in range(self.size)]
        shuffledIndicies = dataIndicies[:]

        self.randomNumberGenerator.shuffle(shuffledIndicies)

        onTimeBatchedInputsAndOutputs = []
        batchedInputsAndOutputs = []

        for dataIndex in shuffledIndicies:
            onTimeBatchedInputsAndOutputs.append((
                self.loadImage(dataIndex),
                self.outputs[dataIndex],
            ))

            if len(onTimeBatchedInputsAndOutputs) == self.batchSize:
                batchedInputsAndOutputs.append(onTimeBatchedInputsAndOutputs)
                onTimeBatchedInputsAndOutputs = []

        if self.appendBatchResidue and len(onTimeBatchedInputsAndOutputs) > 0:
            batchedInputsAndOutputs.append(onTimeBatchedInputsAndOutputs)

        return batchedInputsAndOutputs
