from .dense import Dense
from .layer import Layer
from .blank import Blank

from ..dict import DictEncoder

layersDict = DictEncoder().encodeTypes([
    Dense,
    Layer,
    Blank
])
