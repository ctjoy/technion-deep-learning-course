import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement a CNN. Save the layers in the modules list.
        # The input shape is an image batch: (N, in_channels, H_in, W_in).
        # The output shape should be (N, out_channels, H_out, W_out).
        # You can assume H_in, W_in >= 64.
        # Architecture is up to you, but you should use at least 3 Conv layers.
        # You can use any Conv layer parameters, use pooling or only strides,
        # use any activation functions, use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        """
        We want to get dimension reduction from to
        Layers: 
        input ---> H_in(H), W_in(W) 
        conv1 ---> kernel size: (5,5),S=1 ,P=2, D=1  n_filters:64 
        pool1 ---> H/2, W/2
        batch_norm1
        relu1 
        conv2 ---> kernel size: (5,5),S=2 ,P=2, D=1  n_filters:128 
        pool2 ---> H/4,W/4
        batch_norm2 
        relu2
        conv3 ---> kernel size: (5,5),S=2 ,P=2, D=1  n_filters:256
        pool3 ---> H/8,W/8
        batch_norm3 ---> 
        relu3
        conv4 ---> kernel size: (5,5),S=2 ,P=2, D=1  n_filters:1024
        pool4 ---> H/16,W/16
        batch_norm4 ---> 
        relu4
        #FC1 --> H/16*W/16*1024
        #batch_norm4 --->
        #relu4
        """

        self.filters = [64,128,256]
        in_filters = self.filters.copy()
        in_filters.insert(0, in_channels)
        out_filters = self.filters.copy()
        out_filters.insert(3,out_channels)
        self.kernel_sz = 5
        self.pool_sz = 2
        # H_in = 64
        # W_in = 64
        # H_out = H_in
        # W_out = W_in

        for i, (in_dim, out_dim) in enumerate(zip(in_filters, out_filters)):
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size = self.kernel_sz, padding=2, stride=1))
            modules.append(nn.MaxPool2d(kernel_size = self.pool_sz))
            modules.append(nn.BatchNorm2d(out_dim))
            modules.append(nn.ReLU())

            #H_out = H_out / 2
            #W_out = W_out / 2
        #print("here")

        #n_params = int(W_out * H_out * self.filters[-1])
        #self.reshape(n_params,-1)
       # modules.append(nn.Linear(n_params, out_channels))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO: Implement the "mirror" CNN of the encoder.
        # For example, instead of Conv layers use transposed convolutions,
        # instead of pooling do unpooling (if relevant) and so on.
        # You should have the same number of layers as in the Encoder,
        # and they should produce the same volumes, just in reverse order.
        # Output should be a batch of images, with same dimensions as the
        # inputs to the Encoder were.
        # ====== YOUR CODE: ======
        self.filters = [256, 128, 64]
        self.kernel_sz = 5
        self.unpool_sz = 2
        in_filters = self.filters.copy()
        in_filters.insert(0, in_channels)
        #print(in_filters)
        out_filters = self.filters.copy()
        out_filters.insert(3, out_channels)
        #print(out_filters)

        for i, (in_dim, out_dim) in enumerate(zip(in_filters, out_filters)):
            modules.append(nn.ReLU())
            modules.append(nn.BatchNorm2d(in_dim))
            modules.append(nn.Upsample(scale_factor=self.unpool_sz, mode='bilinear'))
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size=self.kernel_sz, padding=2, stride=1))
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add parameters needed for encode() and decode().
        # ====== YOUR CODE: ======
        #self.hidden_dim = in_size[0] * in_size[1] * in_size[2]
        #self.h_shape =(in_size[0], in_size[1], in_size[2])
        #Get the dimension of the feature output
        device = next(self.parameters()).device
        x = torch.randn(1,*in_size, device=device)
        h = features_encoder(x)
        self.h_shape = h.shape[1:]
        self.hidden_dim = torch.zeros(self.h_shape).numel()


        self.ln_u = torch.nn.Linear(self.hidden_dim, z_dim)
        self.ln_logvar = torch.nn.Linear(self.hidden_dim, z_dim)
        self.ln_rec = torch.nn.Linear(z_dim, self.hidden_dim)
        self.in_size = in_size
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h)//h.shape[0]

    def encode(self, x):
        # TODO: Sample a latent vector z given an input x.
        # 1. Use the features extracted from the input to obtain mu and
        # log_sigma2 (mean and log variance) of the posterior p(z|x).
        # 2. Apply the reparametrization trick.
        # ====== YOUR CODE: ======
        #h = x.reshape(-1,self.hidden_dim)
        h = self.features_encoder(x)
        #print(h.size(0))
        h = h.view(h.size(0), -1)
        mu = self.ln_u(h)
        log_sigma2 = self.ln_logvar(h)
        std = torch.exp(0.5*log_sigma2)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        # ========================

        return z, mu, log_sigma2

    def decode(self, z):
        # TODO: Convert a latent vector back into a reconstructed input.
        # 1. Convert latent to features.
        # 2. Apply features decoder.
        # ====== YOUR CODE: ======
        #h_1 = z.reshape(-1, self.z_dim)
        #h_rec = self.ln_rec(z)
        #x_rec = self.ln_rec(h_1)

        h_rec = self.ln_rec(z)
        h_rec = h_rec.view(h_rec.size(0), *self.h_shape)
        x_rec = self.features_decoder(h_rec)
        # ========================

        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO: Sample from the model.
            # Generate n latent space samples and return their reconstructions.
            # Remember that for the model, this is like inference.
            # ====== YOUR CODE: ======
            for i in range(n):
                data = torch.randn(1, self.z_dim)
                sample = torch.squeeze(VAE.decode(self, data))
                samples.append(sample)
            # ========================
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Pointwise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO: Implement the VAE pointwise loss calculation.
    # Remember that the covariance matrix of the posterior is diagonal.
    # ====== YOUR CODE: ======
    N = x.shape[0]
    MSE_loss = nn.MSELoss()
    data_loss = MSE_loss(xr, x) / x_sigma2
    kldiv_loss = torch.mean(-1 - z_log_sigma2 + torch.pow(z_mu, 2) + z_log_sigma2.exp())
    loss = data_loss + kldiv_loss
    # =======================

    return loss, data_loss, kldiv_loss
