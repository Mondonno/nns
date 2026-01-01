"""
Gradient Checking Tests

Tests to verify that analytical gradients computed by backpropagation
match numerical gradients computed via finite differences.
"""

import math
import pytest

from nns.core.gradient_check import (
    numerical_gradient,
    numerical_gradient_vector,
    relative_error,
    check_gradient,
    check_function_gradient,
    check_layer_gradients,
    check_mse_gradient,
)
from nns.core.functions.linear import LinearFunction
from nns.core.functions.relu import RectifiedLinearFunction
from nns.core.functions.sine import SineFunction
from nns.core.functions.mse import MSEFunction
from nns.core.layers.dense import Dense


class TestNumericalGradient:
    """Test basic numerical gradient computation."""
    
    def test_numerical_gradient_quadratic(self):
        """Test numerical gradient of f(x) = x² at various points."""
        # f(x) = x², f'(x) = 2x
        f = lambda x: x ** 2
        
        test_points = [-2, -1, 0, 1, 2, 3.5]
        for x in test_points:
            numerical = numerical_gradient(f, x)
            analytical = 2 * x
            assert abs(numerical - analytical) < 1e-5, f"Failed at x={x}"
    
    def test_numerical_gradient_cubic(self):
        """Test numerical gradient of f(x) = x³."""
        # f(x) = x³, f'(x) = 3x²
        f = lambda x: x ** 3
        
        test_points = [-2, -1, 0.5, 1, 2]
        for x in test_points:
            numerical = numerical_gradient(f, x)
            analytical = 3 * x ** 2
            assert abs(numerical - analytical) < 1e-4, f"Failed at x={x}"
    
    def test_numerical_gradient_sine(self):
        """Test numerical gradient of f(x) = sin(x)."""
        # f(x) = sin(x), f'(x) = cos(x)
        f = math.sin
        
        test_points = [0, math.pi/4, math.pi/2, math.pi, -math.pi/3]
        for x in test_points:
            numerical = numerical_gradient(f, x)
            analytical = math.cos(x)
            assert abs(numerical - analytical) < 1e-5, f"Failed at x={x}"
    
    def test_numerical_gradient_vector(self):
        """Test numerical gradient of f(x, y) = x² + 2xy + y²."""
        # f(x, y) = x² + 2xy + y²
        # ∂f/∂x = 2x + 2y
        # ∂f/∂y = 2x + 2y
        def f(vec):
            x, y = vec
            return x**2 + 2*x*y + y**2
        
        x, y = 1.0, 2.0
        grad = numerical_gradient_vector(f, [x, y])
        
        expected_dx = 2*x + 2*y  # 6
        expected_dy = 2*x + 2*y  # 6
        
        assert abs(grad[0] - expected_dx) < 1e-5
        assert abs(grad[1] - expected_dy) < 1e-5


class TestRelativeError:
    """Test relative error computation."""
    
    def test_relative_error_identical(self):
        """Identical values should have zero error."""
        error = relative_error(5.0, 5.0)
        assert error == 0.0
    
    def test_relative_error_close(self):
        """Close values should have small error."""
        error = relative_error(1.0, 1.0001)
        assert error < 1e-3
    
    def test_relative_error_near_zero(self):
        """Should handle near-zero values gracefully."""
        error = relative_error(1e-10, 2e-10)
        # Both values are very small, error should be computed safely
        assert math.isfinite(error)
    
    def test_relative_error_different_signs(self):
        """Should compute meaningful error for opposite signs."""
        error = relative_error(1.0, -1.0)
        assert error == 2.0  # |1 - (-1)| / max(1, 1) = 2


class TestFunctionGradients:
    """Test gradient checking for activation functions."""
    
    def test_linear_function_gradient(self):
        """LinearFunction: f(x) = ax + b, f'(x) = a."""
        linear = LinearFunction(coefficient=2.5, bias=1.0)
        
        test_inputs = [-3.0, -1.0, 0.0, 1.0, 3.0, 10.0]
        all_passed, results = check_function_gradient(linear, test_inputs)
        
        assert all_passed, f"Linear gradient check failed: {results}"
    
    def test_relu_function_gradient(self):
        """ReLU: f(x) = max(0, ax), f'(x) = a if x > 0 else 0."""
        relu = RectifiedLinearFunction(coefficient=1.0, bias=0.0)
        
        # Note: Skip x=0 as ReLU is non-differentiable there
        test_inputs = [-3.0, -1.0, -0.1, 0.1, 1.0, 3.0]
        all_passed, results = check_function_gradient(relu, test_inputs)
        
        assert all_passed, f"ReLU gradient check failed: {results}"
    
    def test_relu_with_coefficient_gradient(self):
        """ReLU with custom coefficient."""
        relu = RectifiedLinearFunction(coefficient=2.0, bias=0.5)
        
        test_inputs = [-2.0, -0.5, 0.5, 1.0, 2.0]
        all_passed, results = check_function_gradient(relu, test_inputs)
        
        assert all_passed, f"ReLU coefficient gradient check failed: {results}"
    
    def test_sine_function_gradient(self):
        """SineFunction: f(x) = a*sin(x) + b, f'(x) = a*cos(x)."""
        sine = SineFunction(coefficient=1.0, bias=0.0)
        
        test_inputs = [0.0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi, -math.pi/4]
        all_passed, results = check_function_gradient(sine, test_inputs)
        
        assert all_passed, f"Sine gradient check failed: {results}"
    
    def test_sine_with_coefficient_gradient(self):
        """SineFunction with custom coefficient and bias."""
        sine = SineFunction(coefficient=2.5, bias=0.5)
        
        test_inputs = [0.0, math.pi/4, math.pi/2, math.pi, -math.pi/3]
        all_passed, results = check_function_gradient(sine, test_inputs)
        
        assert all_passed, f"Sine coefficient gradient check failed: {results}"


class TestMSEGradient:
    """Test gradient checking for MSE loss function."""
    
    def test_mse_gradient_basic(self):
        """MSE: L = (y - ŷ)², ∂L/∂y = 2(y - ŷ)."""
        mse = MSEFunction()
        
        test_pairs = [
            (1.0, 0.0),   # pred=1, target=0
            (0.0, 1.0),   # pred=0, target=1  
            (2.0, 2.0),   # pred=target, gradient=0
            (3.5, 1.5),   # arbitrary
            (-1.0, 1.0),  # negative prediction
            (0.5, 0.3),   # small difference
        ]
        
        all_passed, results = check_mse_gradient(mse, test_pairs)
        
        assert all_passed, f"MSE gradient check failed: {results}"
    
    def test_mse_gradient_large_values(self):
        """Test MSE gradient with larger values."""
        mse = MSEFunction()
        
        test_pairs = [
            (100.0, 50.0),
            (-50.0, 50.0),
            (1000.0, 999.0),
        ]
        
        all_passed, results = check_mse_gradient(mse, test_pairs)
        
        assert all_passed, f"MSE gradient check for large values failed: {results}"


class TestDenseLayerGradients:
    """Test gradient checking for Dense layer."""
    
    def test_dense_gradient_linear_activation(self):
        """Test Dense layer gradients with linear activation."""
        linear = LinearFunction(coefficient=1.0, bias=0.0)
        
        dense = Dense(
            inputsCount=2,
            neuronsCount=2,
            activation=linear,
            seed=42,
            weights=[[0.5, 0.3, 0.1], [0.2, 0.4, 0.2]]  # 2 inputs + bias per neuron
        )
        
        # Inputs must be filled (replicated for each neuron)
        inputs = [[1.0, 2.0], [1.0, 2.0]]  # 2 neurons, 2 inputs each
        error_derivatives = [1.0, 1.0]  # ∂L/∂a for each neuron
        
        all_passed, results = check_layer_gradients(
            dense, inputs, error_derivatives, 
            epsilon=1e-5, tolerance=1e-4
        )
        
        assert all_passed, f"Dense layer gradient check failed: {[r for r in results if not r['passed']]}"
    
    def test_dense_gradient_sine_activation(self):
        """Test Dense layer gradients with sine activation."""
        sine = SineFunction(coefficient=1.0, bias=0.0)
        
        dense = Dense(
            inputsCount=2,
            neuronsCount=2,
            activation=sine,
            seed=42,
            weights=[[0.3, 0.2, 0.1], [0.1, 0.3, 0.2]]
        )
        
        # Inputs must be filled (replicated for each neuron)
        inputs = [[0.5, 0.5], [0.5, 0.5]]
        error_derivatives = [1.0, 0.5]
        
        all_passed, results = check_layer_gradients(
            dense, inputs, error_derivatives,
            epsilon=1e-5, tolerance=1e-4
        )
        
        assert all_passed, f"Dense (sine) gradient check failed: {[r for r in results if not r['passed']]}"
    
    def test_dense_gradient_single_neuron(self):
        """Test Dense layer with single neuron."""
        linear = LinearFunction(coefficient=1.0, bias=0.0)
        
        dense = Dense(
            inputsCount=3,
            neuronsCount=1,
            activation=linear,
            seed=42,
            weights=[[0.1, 0.2, 0.3, 0.4]]  # 3 inputs + bias
        )
        
        inputs = [[1.0, 2.0, 3.0]]  # Single neuron, 3 inputs
        error_derivatives = [1.0]
        
        all_passed, results = check_layer_gradients(
            dense, inputs, error_derivatives,
            epsilon=1e-5, tolerance=1e-4
        )
        
        assert all_passed, f"Dense (single neuron) gradient check failed: {[r for r in results if not r['passed']]}"
    
    def test_dense_gradient_relu_positive(self):
        """Test Dense layer with ReLU on positive weighted sums."""
        relu = RectifiedLinearFunction(coefficient=1.0, bias=0.0)
        
        # Use weights that will produce positive weighted sums
        dense = Dense(
            inputsCount=2,
            neuronsCount=2,
            activation=relu,
            seed=42,
            weights=[[0.5, 0.5, 1.0], [0.3, 0.3, 0.5]]  # Positive bias ensures positive sum
        )
        
        # Inputs must be filled (replicated for each neuron)
        inputs = [[1.0, 1.0], [1.0, 1.0]]  # Positive inputs
        error_derivatives = [1.0, 1.0]
        
        all_passed, results = check_layer_gradients(
            dense, inputs, error_derivatives,
            epsilon=1e-5, tolerance=1e-4
        )
        
        assert all_passed, f"Dense (ReLU) gradient check failed: {[r for r in results if not r['passed']]}"


class TestGradientCheckEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_zero_gradient(self):
        """Test when analytical gradient is zero."""
        passed, error, _ = check_gradient(0.0, 1e-10)
        # Very small numerical gradient compared to zero analytical
        assert passed or error < 0.02  # Allow some tolerance for zero case
    
    def test_very_small_gradient(self):
        """Test with very small gradient values."""
        passed, error, _ = check_gradient(1e-8, 1.1e-8, tolerance=0.2)
        assert passed  # 10% relative error is acceptable for tiny values
    
    def test_large_gradient(self):
        """Test with large gradient values."""
        passed, error, _ = check_gradient(1000.0, 1000.001)
        assert passed  # Very small relative error


class TestIntegration:
    """Integration tests for complete gradient checking workflow."""
    
    def test_full_gradient_check_workflow(self):
        """Test complete gradient checking on a simple network configuration."""
        # Setup
        linear = LinearFunction(coefficient=1.0, bias=0.0)
        mse = MSEFunction()
        
        # Create a simple layer
        dense = Dense(
            inputsCount=2,
            neuronsCount=1,
            activation=linear,
            seed=42,
            weights=[[0.5, 0.3, 0.1]]
        )
        
        # Test input
        inputs = [[1.0, 2.0]]
        target = [1.5]
        
        # Forward pass
        filled_inputs, outputs = dense.forwardPass(inputs)
        prediction = outputs[0][0]
        
        # Compute loss gradient (MSE)
        loss_grad = mse.derivative([prediction, target[0]])
        
        # Backward pass
        weight_grads, input_grads = dense.backwardPass(inputs, [loss_grad])
        
        # Verify gradients numerically
        epsilon = 1e-5
        
        for i in range(dense.neuronsCount):
            for j in range(dense.inputsCount + 1):
                grad_idx = i * (dense.inputsCount + 1) + j
                original = dense.weights[i][j]
                
                # f(w + eps)
                dense.weights[i][j] = original + epsilon
                _, out_plus = dense.forwardPass(inputs)
                loss_plus = mse.call([out_plus[0][0], target[0]])
                
                # f(w - eps)
                dense.weights[i][j] = original - epsilon
                _, out_minus = dense.forwardPass(inputs)
                loss_minus = mse.call([out_minus[0][0], target[0]])
                
                dense.weights[i][j] = original
                
                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                analytical = weight_grads[grad_idx]
                
                rel_error = relative_error(analytical, numerical)
                assert rel_error < 1e-4, (
                    f"Gradient mismatch at [{i}][{j}]: "
                    f"analytical={analytical}, numerical={numerical}, error={rel_error}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
