
import os
from .dataset import Dataset

class FileDataset(Dataset):
    def __init__(self, directory, batchSize, appendBatchResidue=False):
        if not os.path.isdir(directory):
            raise ValueError(f"Directory '{directory}' does not exist.")
        self.directory = directory

        self.filePaths = []
        for root, dirs, files in os.walk(directory):
            # Only include files in the top-level and one subdirectory deep
            rel_depth = os.path.relpath(root, directory).count(os.sep)
            if rel_depth > 1:
                continue
            for fname in files:
                file_path = os.path.join(root, fname)
                if os.path.isfile(file_path):
                    self.filePaths.append(file_path)

        self.size = len(self.filePaths)
        print(self.size)

        self.batchSize = batchSize
        self.appendBatchResidue = appendBatchResidue
        self.inputs, self.outputs = self._loadAll()
        super().__init__(self.size, self.inputs, self.outputs, self.batchSize, self.appendBatchResidue)

    def _loadAll(self):
        inputs = []
        outputs = []
        for filePath in self.filePaths:
            inputData, outputData = self.parseFile(filePath)
            inputs.append(inputData)
            outputs.append(outputData)
        return inputs, outputs

    def parseFile(self, filePath):
        raise NotImplementedError("Subclasses must implement parseFile method.")
