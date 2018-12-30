r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.003
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.005
    lr_momentum = 0.0004
    lr_rmsprop = 0.0002
    reg = 0.000005
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.0005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
There appears overfitting when there is no dropout because after 10 epochs
when training accuracy increases, validation accuracy didn't improve or even decrease.
After applying dropout=0.4, overfitting is reduced since the test accuracy increases together
with training accuracy. However, since some random units are not put into the training
session with dropout, the training speed slows down. With the highest dropout rate=0.8, there
stills appears overfitting because when training accuracy increases slowly, test accuracy didn't
improve after 10 epochs.
"""

part2_q2 = r"""
Yes,it is possible that while test loss to increase for a few epochs 
while the test accuracy also increases.
This is easily possible with cross entropy loss function that is sensitive to the
distance between an incorrect prediction and the ground truth. 
You can get 90% of the predictions correct, but if the misses are 
ridiculously far off the mark, your loss value can increase.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
