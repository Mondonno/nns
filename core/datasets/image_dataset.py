from PIL import Image

from ..functions.function import Function
from .file_dataset import FileDataset

class ImageDataset(FileDataset):
    def __init__(self, directory, batchSize, transformFunction=None, appendBatchResidue=False, labelFunction=None, imageShape=None, grayscale=False):
        self.labelFunction: Function = labelFunction
        self.transformFunction: Function = transformFunction
        if imageShape is not None and (not isinstance(imageShape, (tuple, list)) or len(imageShape) < 2):
            raise ValueError("imageShape must be None or a tuple/list with at least two elements.")
        self.imageShape = imageShape
        self.grayscale: bool = grayscale
        super().__init__(directory, batchSize, appendBatchResidue)

    def parseFile(self, filePath):
        """
        Parses an image file and returns the processed image data and optional label data.

        Args:
            filePath (str): Path to the image file.

        Returns:
            tuple: A tuple containing:
                - inputData (list): A 2D list representing the pixel data of the image.
                - outputData (any or None): The label data derived from the file path using the labelFunction, or None if no labelFunction is provided.
        """
        
        # Set mode to 'L' for grayscale images or 'RGB' for color images
        mode = 'L' if self.grayscale else 'RGB'
        with Image.open(filePath) as img:
            img = img.convert(mode)
            if self.imageShape is not None:
                img = img.resize(self.imageShape[:2])
            # Convert image to a list (no normalization here)
            imgArray = list(img.getdata())
            w, h = img.size
            imgArray = [imgArray[i * w:(i + 1) * w] for i in range(h)]
            # Apply transform function if provided (should handle normalization)
            if self.transformFunction is not None:
                imgArray = self.transformFunction.call(imgArray)

        inputData = imgArray
        if self.labelFunction is not None:
            outputData = self.labelFunction.call(filePath)
        else:
            outputData = None
        return inputData, outputData
