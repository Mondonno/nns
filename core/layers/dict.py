from .convolution2d import Convolution2D
from .dense import Dense
from .flatten import Flatten
from .layer import Layer
from .maxpool2d import MaxPooling2D
from .blank import Blank

from ..dict import DictEncoder

layersDict = DictEncoder().encodeTypes([
    Convolution2D,
    Dense,
    Flatten,
    Layer,
    MaxPooling2D,
    Blank
])
