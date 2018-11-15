# CS236605: Deep Learning on Computational Accelerators
# Homework Assignment 1

Faculty of Computer Science, Technion.

## Introduction

In this first homework assignment we'll familiarize ourselves with `PyTorch` as a
general-purpose tensor library with automatic gradient calculation capabilities.
We'll use it to implement some traditional machine-learning algorithms and remind ourselves about
basic concepts such as different data sets and their uses, model hyperparameters, cross-validation,
loss functions and gradient derivation. We'll also familiarize ourselves with other highly important
python machine learning packages such as `numpy`, `sklearn` and `pandas`.

## General Guidelines

- Please read the [getting started page](https://vistalab-technion.github.io/cs236605/assignments/getting-started/)
  on the course website. It explains how to **setup, run and submit** the assignment.
- The text and code cells in these notebooks are intended to guide you through the
  assignment and help you verify your solutions.
  The notebooks **do not need to be edited** at all (unless you wish to play around).
  The only exception is to fill your name(s) in the above cell before submission.
  Please do not remove sections or change the order of any cells.
- All your code (and even answers to questions) should be written in the files
  within the python package corresponding the assignment number (`hw1`, `hw2`, etc).
  You can of course use any editor or IDE to work on these files.

## Contents

- Part 1: Working with data in `PyTorch`
    - Datasets
    - Built-in Datasets and Transforms
    - `DataLoader`s and `Sampler`s
    - Training, Validation and Test Sets
- Part 2: Nearest-neighbor image classification:
    - kNN Classification
    - Cross-validation
- Part 3: Multiclass linear classification
    - Linear Classification
    - Loss Functions
    - Optimizing a Loss Function with Gradient Descent
    - Training the model with SGD
    - Automatic differentiation
- Part 4: Linear Regression
    - Dataset exploration
    - Linear Regression Model
    - Adding nonlinear features
    - Generalization

