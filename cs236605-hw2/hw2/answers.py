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
1. When layer increases, the speed of convergenc slows down and the training and test accuracy also reduces. When
the number of layers is 2 ,the test accuracy achieves the highest which means that fewer parameters can
fit the model better.

2. When L equals 16, the network is not trainable. Due to maxpool layer,if the number of layers is too
many,the width and height of the last layer may be less than 1 and therefore the network will be non-trainable.
There are two suggested ways for reducing such an effect. One is to set the number of layers less than a maximum value
to prevent the width or height of the last layer less than 1.The other one is to get rid off maxpool layer when
the number of layers reach the maximum value.
"""

part3_q2 = r"""
The number of units indicate the complexity of the model. We can see from the graph that when increasing the number of units
of neural network, the test accuracy firstly increase as 32 to 64 and then decrease sharply when it reaches 128 and finally increase when it reaches 256.
When number of units reaches 64, the accuracy is the highest which means that model with such complexity fit the data best.

Compared the result of question 1, we can draw the conclusion that when the number of units increase, the test accuracy firstly
increase because of increased complexity of model and then decrease due to overfitting.
"""

part3_q3 = r"""
When changing both the number of filters and increasing the number of layers, the test accuracy decrease greatly.
When there are 16 layers, due to too many maxpooling layer, the size of the final layer is too small to provide effective
information.
"""


part3_q4 = r"""
We add dropout layer and  layer after each relu layer in our own model.
Dropout layer is used to reduce overfitting as shown in the final result.
"""
# ==============
