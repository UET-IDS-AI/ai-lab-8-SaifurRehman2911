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
    return 1 / (1 + np.exp(-z))

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
    h1 = sigmoid(X @ W1)
    h2 = sigmoid(h1 @ W2)
    y  = sigmoid(h2 @ W3)
    return h1, h2, y

def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.
    Returns:
        dW1, dW2, dW3, loss
    """
    # Binary cross-entropy loss
    loss = -np.mean(label * np.log(y + 1e-8) + (1 - label) * np.log(1 - y + 1e-8))

    # Output layer delta
    delta3 = (y - label) * y * (1 - y)          # shape: same as y
    dW3 = h2.T @ delta3                           # gradient w.r.t. W3

    # Hidden layer 2 delta
    delta2 = (delta3 @ W3.T) * h2 * (1 - h2)
    dW2 = h1.T @ delta2                           # gradient w.r.t. W2

    # Hidden layer 1 delta
    delta1 = (delta2 @ W2.T) * h1 * (1 - h1)
    dW1 = X.T @ delta1                            # gradient w.r.t. W1

    return dW1, dW2, dW3, loss
