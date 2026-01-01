# Neural Networks from Scratch

A comprehensive implementation of neural networks built entirely from scratch in Python, designed for deep learning research and education. This project provides a modular, extensible framework for understanding the inner workings of neural networks through hands-on implementation.

## Overview

This project implements fundamental deep learning components without relying on external libraries like TensorFlow or PyTorch. By building everything from the ground up, it offers insights into the mathematical foundations and algorithmic details of neural network training and inference.

## Architecture

The framework is organized into several key modules:

### Core Components

- **Functions**: Activation functions, loss functions, and utility operations
- **Layers**: Building blocks for neural network architectures
- **Models**: High-level model abstractions and training loops
- **Datasets**: Data loading and preprocessing utilities
- **Optimizers**: Gradient-based optimization algorithms

### Key Features

- **Modular Design**: Each component is independently testable and extensible
- **Backpropagation**: Full automatic differentiation through computational graphs
- **Custom Functions**: Support for user-defined activation and loss functions
- **Multiple Optimizers**: Momentum, Nesterov accelerated gradient, and gradient clipping
- **Data Pipeline**: Flexible dataset handling with transforms and batching
- **Gradient Verification**: Built-in tools for validating gradient computations

## Components

### Functions

The framework includes several activation and loss functions:

- **Activation Functions**: Linear, ReLU, Sine, Sine-squared
- **Loss Functions**: Mean Squared Error (MSE)
- **Special Functions**: Convolution operations, custom learning rate schedules

### Layers

Neural network layers provide the computational building blocks:

- **Dense**: Fully connected layers with configurable activation
- **Convolution2D**: 2D convolutional layers for image processing
- **MaxPool2D**: 2D max pooling for downsampling
- **Flatten**: Reshaping layers for transitioning between convolutional and dense layers

### Models

- **Sequential**: Linear stack of layers with automatic forward and backward passes
- **Custom Models**: Extensible base class for complex architectures

### Optimizers

Gradient-based optimization algorithms:

- **Momentum**: Accelerates convergence in relevant directions
- **Nesterov Accelerated Gradient (NAG)**: Improved momentum variant
- **Gradient Clipping**: Prevents exploding gradients

### Datasets and Transforms

- **Dataset**: Base class for data loading with batching and shuffling
- **File/Image Datasets**: Specialized loaders for different data types
- **Transforms**: Data preprocessing (min-max scaling, etc.)

## Gradient Checking

A critical component for verifying the correctness of analytical gradients computed via backpropagation.

### Mathematical Foundation

#### Numerical Gradient (Centered Difference)

For a function $f(x)$, the derivative at point $x$ is approximated by:

$$\frac{\partial f}{\partial x} \approx \frac{f(x + \varepsilon) - f(x - \varepsilon)}{2\varepsilon}$$

This **centered difference** formula has $O(\varepsilon^2)$ error, much better than one-sided difference $O(\varepsilon)$.

#### For Vectors (Partial Derivatives)

For $f(\mathbf{x})$ where $\mathbf{x} = [x_1, x_2, ..., x_n]$:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + \varepsilon \mathbf{e}_i) - f(\mathbf{x} - \varepsilon \mathbf{e}_i)}{2\varepsilon}$$

where $\mathbf{e}_i$ is the unit vector (1 at position $i$, 0 elsewhere).

#### Relative Error

Compare analytical gradient $g_a$ with numerical gradient $g_n$:

$$\text{relative error} = \frac{|g_a - g_n|}{\max(|g_a|, |g_n|, \varepsilon)}$$

**Interpretation:**
- `< 1e-7`: Excellent ✓
- `< 1e-5`: Good ✓
- `< 1e-3`: Acceptable (for ReLU, softmax)
- `> 1e-3`: Likely bug ✗

## Usage

### Basic Model Training

```python
from core.models.sequential import Sequential
from core.layers.dense import Dense
from core.functions.relu import RectifiedLinearFunction
from core.functions.mse import MSEFunction
from core.functions.linear import LinearFunction

# Define a simple neural network
model = Sequential([
    Dense(2, 4, RectifiedLinearFunction()),
    Dense(4, 1, LinearFunction())
], MSEFunction(), learning_rate_function)

# Train on dataset
model.fit(dataset, epochs=1000)
```

### Custom Functions

```python
from core.functions.function import Function

class CustomActivation(Function):
    def call(self, x):
        return x ** 2  # Custom activation
    
    def derivative(self, x):
        return 2 * x   # Derivative
```

### Gradient Verification

```python
from core.gradient_check import check_function_gradient
from core.functions.sine import SineFunction

sine = SineFunction(coefficient=1.0)
passed, results = check_function_gradient(sine, test_inputs=[0, 0.5, 1.0])
```

## Testing

Comprehensive test suite covering all components:

```bash
# Run all tests
pytest

# Run gradient checking tests
pytest core/test_gradient_check.py -v

# Run layer tests
pytest core/layers/ -v
```

## Practical Notes

1. **Gradient Checking**: Use during development to verify backpropagation implementations. Disable for production training due to computational cost.

2. **Initialization**: Proper weight initialization (Xavier, etc.) is crucial for training stability.

3. **Learning Rates**: Experiment with different schedules for optimal convergence.

4. **Debugging**: The modular design allows testing individual components in isolation.

5. **Performance**: This implementation prioritizes clarity over speed. For production use, consider optimized libraries.

## Project Structure

```
nns/
├── core/                    # Core framework
│   ├── functions/          # Activation and loss functions
│   ├── layers/             # Neural network layers
│   ├── models/             # Model architectures
│   ├── datasets/           # Data handling
│   └── gradient_check.py   # Gradient verification
├── nns/                    # Example applications
├── perceptron/             # Simple perceptron examples
└── docs/                   # Documentation
```