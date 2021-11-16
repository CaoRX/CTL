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

# make a link between two tensors
>>> from CTL.tensor.contract.link import makeLink
>>> shapeA = (3, 4, 5)
>>> shapeB = (5, 4)
>>> a = Tensor(labels = ['a3', 'a4', 'a5'], data = np.ones(shapeA))
>>> b = Tensor(labels = ['b5', 'b4'], data = np.ones(shapeB))
>>> bond = makeLink('a4', 'b4', a, b)
>>> c = a @ b
>>> c
Tensor(shape = (3, 5, 5), labels = ['a3', 'a5', 'b5'])

```

## GPU support
**This feature may be a little unstable since we may not exclude all differences between cupy and numpy by far.**

CTL supports cupy for Nvidia GPU for accelerating heavy tensor contractions. To use GPU, we need the following codes:
```python
import numpy as np
import cupy as cp # cupy is not dependency of CTL, please install manually
import CTL

CTL.setXP(cp) # set the numpy-like library to cupy, and all calculations will be under cupy
```
And then all the calculations below will be applied with cupy. There is an example bin/heavy-example-cupy.py for cupy calculation, compared with bin/heavy-example.py for doing the same job with CPU. You can run
```console
time python bin/heavy-example-cupy.py
time python bin/heavy-example.py
```
to see the difference on running time. The author's PC(GeForce GTX 1060 3GB, AMD® Ryzen 5 1600) shows a 10-time speedup with cupy on "user" time.

Also note that, for tasks that are not heavy, use GPU may not be a good idea: the time to initialize the usage of GPU may be much longer than calculation time. So please choose proper way to run your program with CTL.

## Tensor network contraction example

Run the bin/example.py for a simple example
```console
python bin/example.py
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

## Arbitrary TN contraction example
Run the bin/CATN.py for an Arbitrary TN example [\[1\]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.060503) after finishing the installation:
```console
python bin/CATN.py
```

```python
contractHandmadeTN()
```

This example builds a hand-made tensor network with random initialization, and compared the results of contraction between direct contraction and contraction with the help of MPS. Note that this functionality is a beta version, so may not work well for tensor networks built by users.

```python
squareIsingTest()
```

This example builds a square lattice Ising model of size (4, 4) with free boundary condition, and apply direct diagonalization, tensor network contraction and MPS contraction to calculate the partition function, and the results are compatible with each other.

\[1\] Pan F, Zhou P, Li S, et al. Contracting arbitrary tensor networks: general approximate algorithm and applications in graphical models and quantum circuit simulations\[J\]. Physical Review Letters, 2020, 125(6): 060503.
