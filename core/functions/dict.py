from .crossentropy import CrossEntropyFunction
from .function import Function
from .linear import LinearFunction
from .mse import MSEFunction
from .relu import RectifiedLinearFunction
from .sine import SineFunction
from .sinesq import SineSquaredFunction
from .softmax import SoftmaxFunction

from ..dict import DictEncoder

functionsDict = DictEncoder().encodeTypes([
    Function,
    CrossEntropyFunction,
    LinearFunction,
    MSEFunction,
    RectifiedLinearFunction,
    SineFunction,
    SineSquaredFunction,
    SoftmaxFunction
])
