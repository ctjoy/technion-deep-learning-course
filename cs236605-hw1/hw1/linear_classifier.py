import torch
import cs236605.plot as plot
from torch import Tensor
from torch.utils.data import DataLoader
from collections import namedtuple

from .losses import ClassifierLoss


class LinearClassifier(object):

    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO: Create weights tensor of appropriate dimensions
        # Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        w = torch.empty(n_features, n_classes)
        self.weights = torch.nn.init.normal_(w, mean=0, std=weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO: Implement linear prediction.
        # Calculate the score for each class using the weights and
        # return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x, self.weights)
        y_pred_value, y_pred = torch.max(class_scores, 1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO: calculate accuracy of prediction.
        # Use the predict function above and compare the predicted class
        # labels to the ground truth labels to obtain the accuracy (in %).
        # Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        acc = float(torch.sum(y == y_pred))/float(len(y))
        # ========================

        return acc * 100

    def train(self,
              dl_train: DataLoader,
              dl_valid: DataLoader,
              loss_fn: ClassifierLoss,
              learn_rate=0.1, weight_decay=0.001, max_epochs=100):

        Result = namedtuple('Result', 'accuracy loss')
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print('Training', end='')
        for epoch_idx in range(max_epochs):

            # TODO: Implement model training loop.
            # At each epoch, evaluate the model on the entire training set
            # (batch by batch) and update the weights.
            # Each epoch, also evaluate on the validation set.
            # Accumulate average loss and total accuracy for both sets.
            # The train/valid_res variables should hold the average loss and
            # accuracy per epoch.
            #
            # Don't forget to add a regularization term to the loss, using the
            # weight_decay parameter.

            total_correct = 0
            average_loss = 0

            # ====== YOUR CODE: ======
            mean_acc = 0
            for (x,y) in dl_train:
                y_pred, x_scores = self.predict(x)

                # Compute loss
                loss = loss_fn(x, y, x_scores, y_pred)
                grad = loss_fn.grad()

                # Update weights
                self.weights = self.weights - learn_rate * grad

                average_loss += (loss + weight_decay)
                mean_acc += self.evaluate_accuracy(y, y_pred)

            mean_acc /= len(dl_train)
            average_loss /= len(dl_train)

            train_res.accuracy.append(mean_acc)
            train_res.loss.append(average_loss)

            average_loss = 0
            mean_acc = 0

            for (x,y) in dl_valid:
                y_pred, x_scores = self.predict(x)
                loss = loss_fn(x, y, x_scores, y_pred)

                average_loss += (loss + weight_decay)
                mean_acc += self.evaluate_accuracy(y, y_pred)

            mean_acc /= len(dl_valid)
            average_loss /= len(dl_valid)

            valid_res.accuracy.append(mean_acc)
            valid_res.loss.append(average_loss)

            # ========================
            print('.', end='')

        print('')
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be at the end).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO: Convert the weights matrix into a tensor of images.
        # The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        if has_bias:
            w_images = torch.t(self.weights)[:,:-1].reshape((self.n_classes, *img_shape))
        else:
            w_images = torch.t(self.weights).reshape((self.n_classes, *img_shape))
        # ========================

        return w_images
