from .function import Function
from .linear import LinearFunction
from .mse import MSEFunction
from .relu import RectifiedLinearFunction
from .sine import SineFunction
from .sinesq import SineSquaredFunction

from ..dict import DictEncoder

functionsDict = DictEncoder().encodeTypes([
    Function,
    LinearFunction,
    MSEFunction,
    RectifiedLinearFunction,
    SineFunction,
    SineSquaredFunction
])