"""
AI_stats_lab.py

Neural Networks Lab: 3-Layer Forward Pass and Backpropagation

Implement all functions.
Do NOT change function names.
Do NOT print inside functions.
"""

import numpy as np


def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    y=1/(1+np.exp(-z))
    return y


def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.

    Layer 1:
        h1 = sigmoid(XW1)

    Layer 2:
        h2 = sigmoid(h1W2)

    Output layer:
        y = sigmoid(h2W3)

    Returns:
        h1, h2, y
    """
    h1 = sigmoid(np.dot(X, W1))
    h2 = sigmoid(np.dot(h1, W2))
    y = sigmoid(np.dot(h2, W3))
    return h1, h2, y


def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.

    Returns:
        dW1, dW2, dW3, loss
    """
    m = X.shape[0]  # number of samples
    
    # 1. Binary Cross-Entropy Loss
    # We add a tiny epsilon to avoid log(0)
    loss = -np.mean(label * np.log(y + 1e-9) + (1 - label) * np.log(1 - y + 1e-9))

    # 2. Output layer error (dz3)
    # For Sigmoid + Cross-Entropy, the gradient simplifies to (y - label)
    dz3 = y - label 

    # 3. Layer 2 error (dz2)
    # Propagate back through W3 and apply Sigmoid derivative for h2
    dz2 = np.dot(dz3, W3.T) * (h2 * (1 - h2))

    # 4. Layer 1 error (dz1)
    # Propagate back through W2 and apply Sigmoid derivative for h1
    dz1 = np.dot(dz2, W2.T) * (h1 * (1 - h1))

    # 5. Gradients for weights
    dW3 = np.dot(h2.T, dz3) / m
    dW2 = np.dot(h1.T, dz2) / m
    dW1 = np.dot(X.T, dz1) / m

    return dW1, dW2, dW3, loss
