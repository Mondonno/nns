import struct

from nns.core.datasets.mnist_dataset import MnistDataset, parse_idx1_ubyte, parse_idx3_ubyte


def _write_idx3_file(filePath, images):
    rowsCount = len(images[0])
    columnsCount = len(images[0][0])

    with open(filePath, "wb") as file:
        file.write(struct.pack(">IIII", 2051, len(images), rowsCount, columnsCount))

        for image in images:
            for row in image:
                file.write(bytes(row))


def _write_idx1_file(filePath, labels):
    with open(filePath, "wb") as file:
        file.write(struct.pack(">II", 2049, len(labels)))
        file.write(bytes(labels))


def test_parse_idx_files(tmp_path):
    images = [
        [
            [0, 255],
            [128, 64],
        ],
        [
            [255, 0],
            [32, 16],
        ],
    ]
    labels = [3, 7]

    imagesFilePath = tmp_path / "train-images-idx3-ubyte"
    labelsFilePath = tmp_path / "train-labels-idx1-ubyte"

    _write_idx3_file(imagesFilePath, images)
    _write_idx1_file(labelsFilePath, labels)

    parsedImages = parse_idx3_ubyte(imagesFilePath, normalize=True)
    parsedLabels = parse_idx1_ubyte(labelsFilePath)

    assert len(parsedImages) == 2
    assert parsedImages[0][0][1] == 1.0
    assert parsedImages[0][1][0] == 128 / 255
    assert parsedLabels == labels


def test_mnist_dataset_lazy_batches(tmp_path):
    images = [
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
    ]
    labels = [1, 9]

    imagesFilePath = tmp_path / "train-images-idx3-ubyte"
    labelsFilePath = tmp_path / "train-labels-idx1-ubyte"

    _write_idx3_file(imagesFilePath, images)
    _write_idx1_file(labelsFilePath, labels)

    dataset = MnistDataset(
        imagesFilePath=imagesFilePath,
        labelsFilePath=labelsFilePath,
        batchSize=1,
        limit=2,
        normalize=False,
        oneHot=True,
        lazy=True,
        seed=0,
    )

    batches = dataset.generate()
    flattenedBatches = [sample for batch in batches for sample in batch]
    observedLabels = sorted(output.index(1) for _, output in flattenedBatches)

    assert len(flattenedBatches) == 2
    assert observedLabels == labels
    assert any(inputData == images[0] for inputData, _ in flattenedBatches)
