from random import shuffle

class Dataset():
    def __init__(self, size, inputs, outputs, batchSize, appendBatchResidue = False):
        self.size = size

        self.inputs = inputs
        self.outputs = outputs

        self.batchSize = batchSize
        self.appendBatchResidue = appendBatchResidue

        if(self.batchSize > self.size):
            raise ValueError()
    
    def generate(self):
        dataIndicies = [ i for i in range(self.size) ]
        shuffledIndicies = dataIndicies

        shuffle(shuffledIndicies)

        shuffledInputs = [ self.inputs[i] for i in range(len(shuffledIndicies)) ]
        shuffledOutputs = [ self.outputs[i] for i in range(len(shuffledIndicies)) ]

        onTimeBatchedInputsAndOutputs = []
        batchedInputsAndOutputs = []

        for i in range(1, self.size + 1):
            onTimeBatchedInputsAndOutputs.append((
                shuffledInputs[i - 1],
                shuffledOutputs[i - 1]
            ))

            if i % self.batchSize == 0:
                batchedInputsAndOutputs.append(onTimeBatchedInputsAndOutputs)
                onTimeBatchedInputsAndOutputs = []

        if self.appendBatchResidue and len(onTimeBatchedInputsAndOutputs) > 0:
            batchedInputsAndOutputs.append(onTimeBatchedInputsAndOutputs)

        return batchedInputsAndOutputs