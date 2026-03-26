from random import Random

class Dataset():
    def __init__(self, size, inputs, outputs, batchSize, seed = None, appendBatchResidue = False):
        self.inputs = inputs
        self.outputs = outputs

        if size is None:
            self.size = min(len(self.inputs), len(self.outputs))
        else:
            self.size = size

        if seed is None:
            self.seed = Random().random()
        else:
            self.seed = seed

        self.batchSize = batchSize
        self.appendBatchResidue = appendBatchResidue

        self.randomNumberGenerator = Random(self.seed)

        if(self.batchSize > self.size):
            raise ValueError()
    
    def generate(self):
        dataIndicies = [ i for i in range(self.size) ]
        shuffledIndicies = dataIndicies[:]

        self.randomNumberGenerator.shuffle(shuffledIndicies)

        onTimeBatchedInputsAndOutputs = []
        batchedInputsAndOutputs = []

        for i in range(1, len(shuffledIndicies) + 1):
            shuffledIndex = shuffledIndicies[i - 1]

            onTimeBatchedInputsAndOutputs.append((
                self.inputs[shuffledIndex],
                self.outputs[shuffledIndex]
            ))

            if i % self.batchSize == 0:
                batchedInputsAndOutputs.append(onTimeBatchedInputsAndOutputs)
                onTimeBatchedInputsAndOutputs = []

        if self.appendBatchResidue and len(onTimeBatchedInputsAndOutputs) > 0:
            batchedInputsAndOutputs.append(onTimeBatchedInputsAndOutputs)

        return batchedInputsAndOutputs
