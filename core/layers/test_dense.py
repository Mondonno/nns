import math
import pytest

from nns.core.functions import SineFunction
from nns.core.functions.softmax import SoftmaxFunction
from nns.core.layers.dense import Dense

class DummyActivation:
    def call(self, x):
        return x
    def derivative(self, x):
        return 1

class DummyInitializator:
    def call(self, layer, i, j):
        return 0.5

def test_dense_forward_pass_with_sine():
    activation = SineFunction()
    initializator = DummyInitializator()
    dense = Dense(inputsCount=2, neuronsCount=2, activation=activation, seed=42, initializator=initializator)
    # 2 inputs + bias, all weights 0.5
    inputs = [[1, 2]]
    filled_inputs, outputs = dense.forwardPass(inputs)
    # Each neuron: weighted sum = 1*0.5 + 2*0.5 + 0.5 = 2.0, output = sin(2.0)
    expected_output = math.sin(2.0)
    assert outputs == [[expected_output], [expected_output]]
    assert filled_inputs == [[1, 2], [1, 2]]

def test_dense_forward_pass_basic():
    activation = DummyActivation()
    initializator = DummyInitializator()
    dense = Dense(inputsCount=2, neuronsCount=2, activation=activation, seed=42, initializator=initializator)
    # 2 inputs + bias, all weights 0.5
    inputs = [[1, 2]]
    filled_inputs, outputs = dense.forwardPass(inputs)
    # Each neuron: 1*0.5 + 2*0.5 + 0.5 (bias) = 2.0
    assert outputs == [[2.0], [2.0]]
    assert filled_inputs == [[1, 2], [1, 2]]

def test_dense_weighted_sum():
    activation = DummyActivation()
    initializator = DummyInitializator()
    dense = Dense(inputsCount=3, neuronsCount=1, activation=activation, seed=1, initializator=initializator)
    dense.weights = [[0.1, 0.2, 0.3, 0.4]]  # 3 inputs + bias
    inputs = [[2, 3, 4]]
    result = dense.weightedSum([[2, 3, 4]])
    # 2*0.1 + 3*0.2 + 4*0.3 + 0.4 = 0.2 + 0.6 + 1.2 + 0.4 = 2.4
    assert math.isclose(result[0], 2.4)

def test_dense_derivatives():
    class LinearActivation:
        def call(self, x):
            return x
        def derivative(self, x):
            return 2*x
    activation = LinearActivation()
    initializator = DummyInitializator()
    dense = Dense(inputsCount=1, neuronsCount=1, activation=activation, seed=0, initializator=initializator)
    dense.weights = [[1, 0]]  # 1 input + bias
    inputs = [[3]]
    derivs = dense.derivatives(inputs)
    # weighted sum = 3*1 + 0 = 3, derivative = 2*3 = 6
    assert derivs == [6]

def test_dense_from_dict():
    class DummyActivationType:
        @classmethod
        def fromDict(cls, obj, ref):
            return DummyActivation()
    class DummyInitializatorType:
        @classmethod
        def fromDict(cls, obj, ref):
            return DummyInitializator()
    objectDict = {
        "inputsCount": 1,
        "neuronsCount": 1,
        "activation": {"name": "DummyActivationType"},
        "initializator": {"name": "DummyInitializatorType"},
        "seed": 123,
        "weights": [[0.5, 0.5]]
    }
    additionalDict = {
        "DummyActivationType": DummyActivationType,
        "DummyInitializatorType": DummyInitializatorType
    }
    dense = Dense.fromDict(objectDict, additionalDict)
    assert isinstance(dense, Dense)
    assert dense.weights == [[0.5, 0.5]]
    assert dense.inputsCount == 1
    assert dense.neuronsCount == 1

def test_dense_backward_pass_minimal():
    activation = DummyActivation()
    initializator = DummyInitializator()
    dense = Dense(inputsCount=1, neuronsCount=1, activation=activation, seed=0, initializator=initializator)
    dense.weights = [[1, 0.5]]  # 1 input + bias
    inputs = [[3]]

    weight_derivatives, next_layer_error_derivatives = dense.backwardPass(inputs, [1])

    assert weight_derivatives == [3, 1]
    assert next_layer_error_derivatives == [1]

def test_dense_forward_pass_with_softmax_vector_activation():
    dense = Dense(
        inputsCount=2,
        neuronsCount=2,
        activation=SoftmaxFunction(),
        seed=0,
        weights=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    _, outputs = dense.forwardPass([[2.0, 1.0]])

    output_values = [single_output[0] for single_output in outputs]
    expected_exp_2 = math.exp(2.0)
    expected_exp_1 = math.exp(1.0)
    expected_sum = expected_exp_2 + expected_exp_1

    assert pytest.approx(output_values[0]) == expected_exp_2 / expected_sum
    assert pytest.approx(output_values[1]) == expected_exp_1 / expected_sum

def test_dense_backward_pass_with_softmax_jacobian():
    dense = Dense(
        inputsCount=2,
        neuronsCount=2,
        activation=SoftmaxFunction(),
        seed=0,
        weights=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )

    inputs = [[2.0, 1.0]]
    error_derivatives = [-1.0, 0.0]

    weight_derivatives, next_layer_error_derivatives = dense.backwardPass(inputs, error_derivatives)

    probability_0 = math.exp(2.0) / (math.exp(2.0) + math.exp(1.0))
    probability_1 = 1 - probability_0
    expected_delta_0 = -(probability_0 * (1 - probability_0))
    expected_delta_1 = probability_0 * probability_1

    assert pytest.approx(weight_derivatives) == [
        2.0 * expected_delta_0,
        1.0 * expected_delta_0,
        expected_delta_0,
        2.0 * expected_delta_1,
        1.0 * expected_delta_1,
        expected_delta_1,
    ]
    assert pytest.approx(next_layer_error_derivatives) == [
        expected_delta_0,
        expected_delta_1,
    ]
