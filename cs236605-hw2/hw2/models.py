import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is:

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.

    If dropout is used, a dropout layer is added after every ReLU.
    """
    def __init__(self, in_features, num_classes, hidden_features=(),
                 dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param: Dropout probability. Zero means no dropout.
        """
        self.in_features = in_features
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.dropout = dropout

        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        for i, h in enumerate(hidden_features):
            if i == 0:
                blocks.append(Linear(in_features, h))
            else:
                blocks.append(Linear(hidden_features[i-1], h))

            blocks.append(ReLU())

            if self.dropout != 0:
                blocks.append(Dropout(self.dropout))

        blocks.append(Linear(hidden_features[-1], num_classes))
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        filters = self.filters.copy()
        filters.insert(0, in_channels)

        out_h = in_h
        out_w = in_w

        for i, (in_dim, out_dim) in enumerate(zip(filters, self.filters)):
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), padding=1))
            layers.append(nn.ReLU())

            if (i + 1) % self.pool_every == 0:

                if ((out_w - 2) / 2 + 1) >= 1 or ((out_h - 2) / 2 + 1) >= 1:

                    out_h = (out_h - 2) / 2 + 1
                    out_w = (out_w - 2) / 2 + 1

                    layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        self.classifier_in_features = int(out_h * out_w * self.filters[-1])
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        hidden_dims = self.hidden_dims.copy()
        hidden_dims.insert(0, self.classifier_in_features)

        for in_dim, out_dim in zip(hidden_dims, self.hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        filters = self.filters.copy()
        filters.insert(0, in_channels)

        out_h = in_h
        out_w = in_w

        for i, (in_dim, out_dim) in enumerate(zip(filters, self.filters)):
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), padding=1))
            layers.append(nn.Dropout2d(p=0.5))
            layers.append(nn.ReLU())

            if (i + 1) % self.pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

                out_h = (out_h - 2) / 2 + 1
                out_w = (out_w - 2) / 2 + 1

        self.classifier_in_features = int(out_h * out_w * self.filters[-1])
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        hidden_dims = self.hidden_dims.copy()
        hidden_dims.insert(0, self.classifier_in_features)

        for in_dim, out_dim in zip(hidden_dims, self.hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])

        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq
    # ========================
