# Tensorlib
A simple Tensor library for tensor-product simulation

## Introduction
The tensor-product form of quantum objects plays a central role in the simulation of one-dimensional quantum many-body systems.

## Requirement
* [BLAS](http://www.netlib.org/blas) - Basic Linear Algebra Subprograms
* [LAPACKE](http://www.netlib.org/lapack/lapacke.html) - The LAPACKE C Interface to LAPACK

## Build and Test
* In source root
```
$ make
```

* Run test itebd
```
./itebd 
Input xfield: 0.02
Input zfield: 0.03
```

## Reference
* [The density-matrix renormalization group in the age of matrix product states](https://arxiv.org/abs/1008.3477)
* [The iTEBD algorithm beyond unitary evolution](https://arxiv.org/abs/0711.3960)
