# Mnist Dataset From Scratch (No Pytorch, TensorFlow or keras)
# MNIST Digit Classifier from Scratch

This project implements a simple feedforward neural network from scratch to classify handwritten digits from the MNIST dataset. It avoids using high-level machine learning libraries like PyTorch or TensorFlow, relying instead on Python and NumPy. The goal is to show a clear understanding of how neural networks work under the hood.

---

## 🧠 Project Overview

The neural network is trained to classify digits (0–9) from 28x28 grayscale images. Each image is flattened into a 784-dimensional vector (28 × 28) and passed through a fully connected feedforward neural network.

---

## 🏗️ Neural Network Architecture

- **Input Layer:** 784 neurons (one per pixel)
- **Hidden Layer(s):** Configurable size (e.g., 128 or 64 neurons)
- **Output Layer:** 10 neurons (digits 0 through 9)

Each layer is fully connected to the next.

---

## 🔁 Activation Functions

### ReLU (Rectified Linear Unit)

Used in hidden layers to introduce non-linearity:

```
ReLU(z) = max(0, z)
```

Its derivative (used in backpropagation):

```
ReLU'(z) = 1 if z > 0 else 0
```

### Softmax

Used at the output layer to produce a probability distribution over the digit classes:

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)
```

---

## 🔄 Forward Propagation

1. **Linear Transformation:**

```
z(l) = W(l) * a(l-1) + b(l)
```

2. **Activation:**

```
a(l) = ReLU(z(l))    # for hidden layers
a(L) = softmax(z(L)) # for output layer
```

This process continues layer by layer until the final output is produced.

---

## 📉 Loss Function: Cross-Entropy

The cross-entropy loss measures how well the predicted output matches the true label:

```
L = -Σ y_i * log(ŷ_i)
```

Where:
- `y_i` is the true label (one-hot encoded)
- `ŷ_i` is the predicted probability for class `i`

---

## 🔁 Backpropagation

Used to calculate gradients of the loss with respect to the weights and biases.

1. **Output Error:**

```
δ(L) = ŷ - y
```

2. **Backpropagation Through Hidden Layers:**

```
δ(l) = (W(l+1)^T) * δ(l+1) * ReLU'(z(l))
```

3. **Gradient Calculation:**

```
∂L/∂W(l) = δ(l) * (a(l-1))^T
∂L/∂b(l) = δ(l)
```

---

## 🧮 Parameter Updates (Gradient Descent)

Weights and biases are updated using gradient descent:

```
W(l) := W(l) - η * ∂L/∂W(l)
b(l) := b(l) - η * ∂L/∂b(l)
```

Where `η` is the learning rate.

---

## 🏃‍♂️ Training Process

The model is trained over several epochs. After each epoch, the model's performance is evaluated (typically using accuracy on a validation set) to track learning progress and check for overfitting.

---

## ✅ Evaluation Metric

- **Accuracy:** The percentage of correctly classified digits.

---

## 🎯 Summary

This project showcases a full neural network implementation built from the ground up using only NumPy and Python. Every core concept — forward propagation, activation functions (ReLU & Softmax), loss calculation, backpropagation, and weight updates — is implemented manually to give an inside look into how deep learning works under the hood.
