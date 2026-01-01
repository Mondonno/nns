"""
Gradient Checking Utilities

Numerical gradient checking to verify analytical gradients computed by backpropagation.
Uses the centered difference formula for higher accuracy:

    ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)

This provides O(ε²) error compared to O(ε) for one-sided difference.
"""

import math
import copy
from typing import Callable, List, Tuple, Optional


def numerical_gradient(
    func: Callable[[float], float],
    x: float,
    epsilon: float = 1e-5
) -> float:
    """
    Compute numerical gradient of a scalar function at point x.
    
    Uses centered difference formula:
        ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)
    
    Args:
        func: Scalar function f(x) -> y
        x: Point at which to compute gradient
        epsilon: Small perturbation value (default: 1e-5)
    
    Returns:
        Numerical approximation of df/dx at x
    """
    f_plus = func(x + epsilon)
    f_minus = func(x - epsilon)
    return (f_plus - f_minus) / (2 * epsilon)


def numerical_gradient_vector(
    func: Callable[[List[float]], float],
    x: List[float],
    epsilon: float = 1e-5
) -> List[float]:
    """
    Compute numerical gradient of a function with vector input.
    
    For each component x_i:
        ∂f/∂x_i ≈ (f(x + ε*e_i) - f(x - ε*e_i)) / (2ε)
    
    Args:
        func: Function f(x) -> scalar where x is a vector
        x: Point (vector) at which to compute gradient
        epsilon: Small perturbation value
    
    Returns:
        Gradient vector [∂f/∂x_0, ∂f/∂x_1, ..., ∂f/∂x_n]
    """
    gradient = []
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        
        grad_i = (f_plus - f_minus) / (2 * epsilon)
        gradient.append(grad_i)
    
    return gradient


def relative_error(
    analytical: float,
    numerical: float,
    epsilon: float = 1e-8
) -> float:
    """
    Compute relative error between analytical and numerical gradients.
    
    Uses formula that handles near-zero values:
        error = |a - n| / max(|a|, |n|, ε)
    
    Args:
        analytical: Analytically computed gradient
        numerical: Numerically computed gradient
        epsilon: Small value to prevent division by zero
    
    Returns:
        Relative error (should be < 1e-5 for correct gradients)
    """
    diff = abs(analytical - numerical)
    denom = max(abs(analytical), abs(numerical), epsilon)
    return diff / denom


def check_gradient(
    analytical: float,
    numerical: float,
    tolerance: float = 1e-5,
    name: str = ""
) -> Tuple[bool, float, str]:
    """
    Check if analytical gradient matches numerical gradient.
    
    Args:
        analytical: Analytically computed gradient
        numerical: Numerically computed gradient
        tolerance: Maximum acceptable relative error
        name: Optional name for error messages
    
    Returns:
        Tuple of (passed, relative_error, message)
    """
    error = relative_error(analytical, numerical)
    passed = error < tolerance
    
    if passed:
        message = f"✓ {name}: rel_error={error:.2e}" if name else f"✓ rel_error={error:.2e}"
    else:
        message = (
            f"✗ {name}: analytical={analytical:.6f}, numerical={numerical:.6f}, "
            f"rel_error={error:.2e} > {tolerance:.0e}"
        ) if name else (
            f"✗ analytical={analytical:.6f}, numerical={numerical:.6f}, "
            f"rel_error={error:.2e} > {tolerance:.0e}"
        )
    
    return passed, error, message


def check_function_gradient(
    func,
    test_inputs: List[float],
    epsilon: float = 1e-5,
    tolerance: float = 1e-5
) -> Tuple[bool, List[dict]]:
    """
    Check gradient implementation of a Function class.
    
    Verifies that func.derivative(x) matches numerical gradient of func.call(x).
    
    Args:
        func: Function instance with call() and derivative() methods
        test_inputs: List of input values to test
        epsilon: Perturbation for numerical gradient
        tolerance: Maximum acceptable relative error
    
    Returns:
        Tuple of (all_passed, list_of_results)
    """
    results = []
    all_passed = True
    
    for x in test_inputs:
        analytical = func.derivative(x)
        numerical = numerical_gradient(func.call, x, epsilon)
        
        passed, error, message = check_gradient(analytical, numerical, tolerance, f"x={x}")
        
        results.append({
            'input': x,
            'analytical': analytical,
            'numerical': numerical,
            'relative_error': error,
            'passed': passed,
            'message': message
        })
        
        if not passed:
            all_passed = False
    
    return all_passed, results


def check_layer_gradients(
    layer,
    inputs: List[List[float]],
    error_derivatives: List[float],
    epsilon: float = 1e-5,
    tolerance: float = 1e-4
) -> Tuple[bool, List[dict]]:
    """
    Check gradient computation of a layer using numerical differentiation.
    
    For each weight w_ij, computes:
        numerical: ∂L/∂w_ij ≈ (L(w + ε) - L(w - ε)) / (2ε)
        analytical: from layer.backwardPass()
    
    Args:
        layer: Layer instance with forwardPass() and backwardPass()
        inputs: Input to the layer (should be in filled format for multi-neuron layers)
        error_derivatives: Error derivatives from next layer (∂L/∂a)
        epsilon: Perturbation for numerical gradient
        tolerance: Maximum acceptable relative error
    
    Returns:
        Tuple of (all_passed, list_of_results)
    """
    results = []
    all_passed = True
    
    # Get analytical gradients using filled inputs
    analytical_grads, _ = layer.backwardPass(inputs, error_derivatives)
    
    # Compute numerical gradients for each weight
    grad_idx = 0
    
    for i in range(layer.neuronsCount):
        for j in range(layer.inputsCount + 1):
            original_weight = layer.weights[i][j]
            
            # Compute f(w + epsilon)
            layer.weights[i][j] = original_weight + epsilon
            filled_inputs_plus, outputs_plus = layer.forwardPass(inputs)
            loss_plus = sum(
                outputs_plus[k][0] * error_derivatives[k] 
                for k in range(layer.neuronsCount)
            )
            
            # Compute f(w - epsilon)
            layer.weights[i][j] = original_weight - epsilon
            filled_inputs_minus, outputs_minus = layer.forwardPass(inputs)
            loss_minus = sum(
                outputs_minus[k][0] * error_derivatives[k] 
                for k in range(layer.neuronsCount)
            )
            
            # Restore original weight
            layer.weights[i][j] = original_weight
            
            # Numerical gradient
            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            analytical = analytical_grads[grad_idx]
            
            passed, error, message = check_gradient(
                analytical, numerical, tolerance, 
                f"weight[{i}][{j}]"
            )
            
            results.append({
                'weight_idx': (i, j),
                'analytical': analytical,
                'numerical': numerical,
                'relative_error': error,
                'passed': passed,
                'message': message
            })
            
            if not passed:
                all_passed = False
            
            grad_idx += 1
    
    return all_passed, results


def check_mse_gradient(
    mse_func,
    test_pairs: List[Tuple[float, float]],
    epsilon: float = 1e-5,
    tolerance: float = 1e-5
) -> Tuple[bool, List[dict]]:
    """
    Check gradient of MSE loss function.
    
    MSE: L = (y - ŷ)²
    ∂L/∂y = 2(y - ŷ)
    
    Args:
        mse_func: MSE function instance
        test_pairs: List of (prediction, target) pairs
        epsilon: Perturbation for numerical gradient
        tolerance: Maximum acceptable relative error
    
    Returns:
        Tuple of (all_passed, list_of_results)
    """
    results = []
    all_passed = True
    
    for pred, target in test_pairs:
        # Analytical gradient
        analytical = mse_func.derivative([pred, target])
        
        # Numerical gradient with respect to prediction
        def loss_func(p):
            return mse_func.call([p, target])
        
        numerical = numerical_gradient(loss_func, pred, epsilon)
        
        passed, error, message = check_gradient(
            analytical, numerical, tolerance, 
            f"(pred={pred}, target={target})"
        )
        
        results.append({
            'prediction': pred,
            'target': target,
            'analytical': analytical,
            'numerical': numerical,
            'relative_error': error,
            'passed': passed,
            'message': message
        })
        
        if not passed:
            all_passed = False
    
    return all_passed, results


def print_gradient_check_results(
    name: str,
    all_passed: bool,
    results: List[dict],
    verbose: bool = True
) -> None:
    """
    Pretty print gradient checking results.
    
    Args:
        name: Name of the component being tested
        all_passed: Whether all checks passed
        results: List of result dictionaries
        verbose: Whether to print individual results
    """
    status = "PASSED ✓" if all_passed else "FAILED ✗"
    print(f"\n{'='*60}")
    print(f"Gradient Check: {name}")
    print(f"Status: {status}")
    print(f"{'='*60}")
    
    if verbose:
        for r in results:
            print(r['message'])
    
    if not all_passed:
        failed_count = sum(1 for r in results if not r['passed'])
        print(f"\nFailed {failed_count}/{len(results)} checks")
    else:
        print(f"\nAll {len(results)} checks passed")
