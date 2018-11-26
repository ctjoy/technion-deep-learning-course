r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

K is the number of nearest points to choose for voting the class of a candidate point.

When K increases, the generalization effect will firstly improve due to
more classification information is added. After K reaches a threshold, the classification accuracy will reach a
highest value. When K continues to increase, the classification accuracy will decrease due to too much redundant
information.

The reason is that on the one hand, when K is too small such as 1, the classification result of one point is only
determined by its nearest point. The result is unreliable because it is always disturbed by the noise of data.
On the other hand, when K is too large, to the extreme, such as n-1 (n is the number of all points), this will
diminish the effect of KNN classifier because the predicted label of every point is the same class which
has the most number of points. In short, when K increases, the generalization effect will firstly improve due to
more classification information is added. After K reaches a threshold, the classification accuracy will reach a
highest value. When K continues to increase, the classification accuracy will decrease due to too much redundant
information.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The basic principle of loss function is that the correct class
for each image to a have a score higher than the incorrect
classes by some fixed margin $\Delta$. Therefore, $\Delta$ is
used only to ensure that this margin is bigger than 0. It ensures that
even if the score of the correct class is
the same as the score of an incorrect class for an image, there
will still be penalty. So, as long as the $\Delta$ is bigger than
0, it doesn't matter how much it is.

"""

part3_q2 = r"""
**Your answer:**
We can actually interpret that the model is actually learning the
weights of input data. During the identification process, each image
is reshaped into one dimension and do the inner product with the weights.
The result is a one-dimensional matrix whose every column corresponds to
the possibility of classifying the digit into a specific class.

When there is a rotation or translation of the test image , reshaped
form of the image can't predict the class based on the trained weights well
and thus results in a misclassification problem.

The same thing between SVM and KNN is that they both use training data to train
the parameters of the model and get accuracy from test data which is different
from the training data.

The difference is that KNN uses L2 norm as the metric and predicts the class based on
argmax of k-nearest neighbor points of training set. while SVM use the probability
of linear transformation of training data as the metric and predict the class based on
the weights obtained.


"""

part3_q3 = r"""
**Your answer:**

1. We can see that the convergence is fast and the validation loss and training loss decrease
 in a same path. So, the learning rate is good. If the learning rate is too high, the training loss
 will decrease very fast and validation loss will decrease slowly. If the learning rate is too low,
 the training loss and validation loss will both decrease slowly.

2. When it approaches convergence, the validation accuracy is lower than training accuracy, but the difference is
not too much. It means that there appears slightly overfitting to the training set.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is that all of the residual values are zero.

The fitness of the trained model based on cv is better than another based on top-5 features
selected by hand. The reason is that there is a higher proportion of data with residual values closer to 0
based on model after cv and it can also be confirmed by the cv's mse is lower than model of 5 hand-selected
features.

"""

part4_q2 = r"""

1.The parameter: $\lambda$ is the regularization term for controlling the optimization loss.
Given the same number of $\lambda$, using a 'logspace' to search $\lambda$ can reach a higher
range of values compared with 'linspace', which fasters the speed of finding the best parameters.

2.number of fit = number of $\lambda$ * number of degree * k_folds = 20 * 4 *3 = 240

"""

# ==============
