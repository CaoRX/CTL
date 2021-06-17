# CTL
![Python package](https://github.com/CaoRX/CTL/actions/workflows/python-package.yml/badge.svg)

A tensor library for tensor network research

## Installation

```console
git clone https://github.com/CaoRX/CTL && cd CTL
pip install .
```

## Simple usage
```python
>>> from CTL.tensor.tensor import Tensor 
>>> import numpy as np

# create a tensor(with default labels from a to z)
>>> Tensor(data = np.zeros((3, 3)))
Tensor(shape = (3, 3), labels = ['a', 'b'])

# create a tensor with given labels
>>> Tensor(data = np.zeros((3, 3)), labels = ['up', 'down'])
Tensor(shape = (3, 3), labels = ['up', 'down'])
```

## Tensor network contraction example

Run the example/example.py for a simple example
```console
python example/example.py
```

For the functions in this example file:

```python
simplestExample()
```
A simple example about how to use FiniteTensorNetwork to create and contract own networks.

```python
HOTRGImpurityExample(beta)
```
An example that calculates the magnet moment of square lattice Ising model, with impurity tensor techniques, in no more than 20 lines. Also compared with exact results to show the correctness.
