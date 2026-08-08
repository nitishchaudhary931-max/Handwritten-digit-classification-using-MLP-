# Handwritten Digit Recognition Using MLP From Scratch

A fully connected **Multi-Layer Perceptron (MLP)** implemented from scratch using **NumPy** for handwritten digit classification on the MNIST dataset.

The project focuses on understanding the mathematical and algorithmic foundations of neural networks rather than relying on high-level deep learning frameworks such as PyTorch or TensorFlow.

---

## 🚀 Project Highlights

* MLP implemented completely from scratch using NumPy
* User-defined network architecture
* Forward propagation
* ReLU activation
* Softmax output layer
* Cross-entropy loss
* Backpropagation implemented from scratch
* Numerical gradient checking
* Mini-batch training
* L2 regularization
* SGD optimizer
* Adam optimizer implemented from scratch
* Training and validation monitoring
* Confusion matrix analysis
* Precision, recall and F1-score
* Per-class accuracy
* Visualization of correctly and incorrectly classified digits
* Controlled hyperparameter experiments

---

## 🧠 Model Architecture

The network architecture is provided as a list where each element represents the number of neurons in a layer.

For example:

```python
architecture = [784, 128, 64, 10]
```

represents:

```text
Input Layer       : 784 neurons
Hidden Layer 1    : 128 neurons
Hidden Layer 2    : 64 neurons
Output Layer      : 10 neurons
```

The implementation is generalized, so architectures such as:

```python
[784, 64, 10]
```

or

```python
[784, 256, 128, 64, 10]
```

can also be used without modifying the underlying MLP implementation.

---

## 📊 Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images of handwritten digits from `0` to `9`.

Each image has dimensions:

```text
28 × 28 pixels
```

The images are flattened before being passed to the MLP:

```text
28 × 28 → 784 features
```

The output layer contains 10 neurons corresponding to the ten digit classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

### Preprocessing

The input pipeline includes:

1. Loading the dataset
2. Flattening the images
3. Normalizing pixel values
4. Splitting the data into training, validation and test sets
5. One-hot encoding the target labels

---

## ⚙️ Neural Network Pipeline

```text
MNIST Dataset
      │
      ▼
Data Preprocessing
      │
      ├── Flatten 28×28 → 784
      ├── Normalize pixels
      └── One-hot encode labels
      │
      ▼
Parameter Initialization
      │
      ▼
Forward Propagation
      │
      ├── Linear Transformation
      ├── ReLU Activation
      └── Softmax Output
      │
      ▼
Cross-Entropy Loss
      │
      ├── L2 Regularization
      │
      ▼
Backpropagation
      │
      ▼
Gradient Checking
      │
      ▼
Mini-Batch Optimization
      │
      ├── SGD
      └── Adam
      │
      ▼
Model Evaluation
      │
      ├── Accuracy
      ├── Confusion Matrix
      ├── Precision / Recall / F1
      └── Error Analysis
```

---

## 🔢 Forward Propagation

For each layer:

[
Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}
]

For hidden layers, ReLU is used:

[
A^{(l)} = \max(0,Z^{(l)})
]

The output layer uses Softmax:

[
P(y=k|x)
========

\frac{e^{z_k}}
{\sum_j e^{z_j}}
]

This produces a probability distribution over the 10 digit classes.

---

## 📉 Loss Function

The primary loss function is categorical cross-entropy:

[
L_{CE}
======

-\frac{1}{m}
\sum_{i=1}^{m}
\sum_{k=1}^{10}
y_{ik}\log(\hat y_{ik})
]

L2 regularization can additionally be applied to the weights:

[
L =
L_{CE}
+
\frac{\lambda}{2m}
\sum_l ||W^{(l)}||_F^2
]

Bias parameters are not regularized.

---

## 🔄 Backpropagation

The gradients are derived and implemented manually using the chain rule.

For the Softmax + Cross-Entropy output layer:

[
\frac{\partial L}{\partial Z}
=============================

\hat{Y}-Y
]

For a weight matrix:

[
\frac{\partial L}{\partial W}
=============================

\frac{1}{m}
A^T\frac{\partial L}{\partial Z}
+
\frac{\lambda}{m}W
]

The gradients are then propagated backward through all hidden layers.

---

## ✅ Gradient Checking

To verify the correctness of the analytical backpropagation implementation, numerical gradients are calculated using finite differences:

[
\frac{\partial L}{\partial\theta}
\approx
\frac{
L(\theta+\epsilon)
------------------

L(\theta-\epsilon)
}{
2\epsilon
}
]

The analytical and numerical gradients are compared using relative error:

[
\text{Difference}
=================

\frac{
||g_{analytical}-g_{numerical}||
}{
||g_{analytical}||
+
||g_{numerical}||
}
]

This provides an independent verification that the backpropagation implementation is mathematically correct.

---

## 🚀 Optimizers

Two optimizers are implemented from scratch.

### 1. Mini-Batch SGD

Parameters are updated according to:

[
\theta_{t+1}
============

## \theta_t

\eta\nabla_\theta L
]

where:

* (\eta) = learning rate
* (\nabla_\theta L) = gradient of the loss

### 2. Adam

Adam maintains exponentially weighted estimates of the first and second moments of the gradients:

[
m_t =
\beta_1m_{t-1}
+
(1-\beta_1)g_t
]

[
v_t =
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
]

followed by bias correction and parameter updates.

The implementation does not use an external optimizer from PyTorch, TensorFlow, or another deep learning framework.

---

## 📈 Model Evaluation

The trained model is evaluated using:

### Overall Accuracy

[
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
]

### Confusion Matrix

The confusion matrix identifies which handwritten digits are most frequently confused with one another.

### Classification Metrics

The project reports:

* Precision
* Recall
* F1-score
* Support

for each digit from `0` to `9`.

### Per-Class Accuracy

Accuracy is also calculated separately for every digit class.

This helps identify classes that are intrinsically more difficult for the model.

---

## 🔍 Error Analysis

Incorrect predictions are extracted from the test set and visualized.

For each misclassified image, the visualization displays:

```text
True Label
Predicted Label
```

This allows qualitative analysis of model errors and provides insight into ambiguous handwritten samples.

---

## 🧪 Experiments

The project can be extended with controlled experiments involving:

### Network Architecture

```text
[784, 64, 10]
[784, 128, 10]
[784, 128, 64, 10]
[784, 256, 128, 64, 10]
```

### Learning Rate

```text
0.001
0.01
0.1
```

### Batch Size

```text
32
64
128
```

### L2 Regularization

```text
0
1e-4
1e-3
```

### Optimizer

```text
SGD
Adam
```

The validation set should be used to select the best configuration. The test set should be reserved for the final evaluation.

---

## 📁 Repository Structure

A recommended repository structure is:

```text
MLP-Handwritten-Digit-Recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── MLP_MNIST.ipynb
│
├── src/
│   ├── data.py
│   ├── activations.py
│   ├── initialization.py
│   ├── forward.py
│   ├── backward.py
│   ├── loss.py
│   ├── optimizers.py
│   ├── gradient_check.py
│   ├── evaluation.py
│   └── mlp.py
│
└── results/
    ├── training_curves/
    ├── confusion_matrix/
    └── misclassified_digits/
```

If the project is currently implemented entirely inside a notebook, you can initially use:

```text
MLP-Handwritten-Digit-Recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
└── MLP_MNIST.ipynb
```

and refactor it into `src/` later.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/MLP-Handwritten-Digit-Recognition.git
```

Navigate into the project:

```bash
cd MLP-Handwritten-Digit-Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

If using the Jupyter notebook:

```bash
jupyter notebook
```

Then open:

```text
MLP_MNIST.ipynb
```

Run the notebook cells sequentially.

---

## 🧰 Technologies Used

* **Python**
* **NumPy** — numerical computation and MLP implementation
* **Pandas** — data handling
* **Matplotlib** — visualization
* **Scikit-learn** — evaluation utilities
* **Jupyter Notebook** — experimentation

No high-level neural-network framework is used to implement the MLP.

---

## 🎯 Learning Objectives

This project was developed to understand the internal mechanics of neural networks, including:

* How fully connected neural networks perform forward propagation
* How gradients are derived using the chain rule
* How backpropagation trains a neural network
* Why gradient checking is useful
* How regularization affects optimization
* How mini-batch training works
* How SGD and Adam differ
* How network architecture affects performance
* How to diagnose classification errors

---

## 🔮 Future Improvements

Possible extensions include:

* Dropout regularization
* Learning-rate scheduling
* Additional activation functions
* Hyperparameter optimization
* Model serialization
* Interactive handwritten-digit prediction
* Web deployment
* Comparison with a PyTorch implementation

---

## 👤 Author

**Nitish Chaudhary**

This project was developed as a from-scratch implementation of a neural network to strengthen understanding of the mathematical foundations and practical optimization of Multi-Layer Perceptrons.

