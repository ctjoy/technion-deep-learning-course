# CS236605: Deep Learning on Computational Accelerators
# Homework Assignment 2

Faculty of Computer Science, Technion.

## Introduction

In this assignment we'll create a from-scratch implementation of two fundemental deep learning concepts: the backpropagation algorithm and stochastic gradient descent-based optimizers.
Following that we will focus on convolutional networks.
We'll use PyTorch to create our own network architectures and train them using GPUs on the course servers, and we'll conduct architecture experiments to determine the the effects of different architectural decisions on the performance of deep networks.

## General Guidelines

- Please read the [getting started page](https://vistalab-technion.github.io/cs236605/assignments/getting-started/) on the course website. It explains how to **setup, run and submit** the assignment.
- Please read the [course servers usage guide](https://vistalab-technion.github.io/cs236605/assignments/hpc-servers/). It explains how to use and run your code on the course servers to benefit from training with GPUs.
- The text and code cells in these notebooks are intended to guide you through the
  assignment and help you verify your solutions.
  The notebooks **do not need to be edited** at all (unless you wish to play around).
  The only exception is to fill your name(s) in the above cell before submission.
  Please do not remove sections or change the order of any cells.
- All your code (and even answers to questions) should be written in the files
  within the python package corresponding the assignment number (`hw1`, `hw2`, etc).
  You can of course use any editor or IDE to work on these files.

## Contents

- Part1: Backpropagation
    - Comparison with PyTorch
    - Block Implementations
    - Building Models
- Part 2: Optimization and Training:
    - Implementing Optimization Algorithms
    - Vanilla SGD with Regularization
    - Training
    - Momentum
    - RMSProp
    - Dropout Regularization
    - Questions
- Part 3: Convolutional Architectures
    - Convolutional layers and networks
    - Building convolutional networks with PyTorch
    - Experimenting with model architectures
    - Questions
