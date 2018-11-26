r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Increasing k will generally make it better for unseen data, but it won't always be better. For example, when k reaches the same dimension as the data point, the unseen data will all be predicted to the majority class of the sample data. As for uping to what point, you need to use k fold method.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The function of Δ (delta) is that the SVM loss function wants the score of the correct class yi to be larger than the incorrect class scores by at least by that. It control the tradeoff between the data loss and the regularization loss in the objective. The key to understanding this is that the magnitude of the weights W has direct effect on the scores (and hence also their differences): As we shrink all values inside W the score differences will become lower, and as we scale up the weights the score differences will all become higher. Therefore, the exact value of the margin between the scores (e.g. Δ=1, or Δ=100) is in some sense meaningless because the weights can shrink or stretch the differences arbitrarily. Hence, the only real tradeoff is how large we allow the weights to grow (through the regularization strength λ).

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

1. Good. If the learning rate is too small, the gradient descent can be too slow, which didn't show in the experiment. If the learning rate is too large, the gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

2. Slightly overfitted to the training set. Because the training accuracy can be above 90%, but the testing accuracy is just 87.3%.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

1. As we can see the yellow region of weight visualization represent the white color in the mnist dataset, and the blue and green region learn the black in the mnist. 7 and 9 are really similar in the weight, while 5 and 6 are not so clear. Therefore some scrawled test image were misclassified.

2. 

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
