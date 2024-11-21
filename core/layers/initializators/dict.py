from ...dict import DictEncoder
from .xavier import XavierInitializatorFunction

initializatorsDict = DictEncoder().encodeTypes([
    XavierInitializatorFunction
])