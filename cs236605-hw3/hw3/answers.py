r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_generation_params():
    start_seq = ""
    temperature = .0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**


"We will split our corpus into shorter sequences of length S chars. Each sample we provide
our model with will therefore be a tensor of shape (S,V) where V is the embedding dimension. 
Our model will operate sequentially on each char in the sequence." Because if training on the 
whole dataset, the dataset is just 1 sample which doesn't contain enough information for the network to train.

"""

part1_q2 = r"""
**Your answer:**


The memory longer than the sequence length depends on the  hidden layers. The past information is stored in hidden layer
to generate later character and some information therefore keeps long and pass forward.

"""

part1_q3 = r"""
**Your answer:**


The reason for not shuffling is because of time dependence of the original text. The former texts have impact
on the later texts. If shuffling while training, such time dependence will be missed.

"""

part1_q4 = r"""
**Your answer:**


1.Temperature sampling works by increasing the probability of the most likely words before sampling. 
In order to make RNN sensitive to more samples at the beginning, we need to contain the information of 
the most likely predicted label of each sample. For T=1, the freezing function is just the identity function. 
The lower the temperature, the more expected rewards affect the probability.

2.For high temperatures (τ→∞), all samples have nearly the same probability. As a result, the generated
text becomes more diverse and displays greater linguistic variety.

3.For a low temperature (τ→0+), the probability of the sample with the highest 
expected reward tends to 1. As a result, the generated text is grammatically correct.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0,
        h_dim=0, z_dim=0, x_sigma2=0,
        learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 128
    hypers['z_dim'] = 128
    hypers['x_sigma2'] = 0.9
    hypers['learn_rate'] = 0.001
    hypers['betas'] = (0.9,0.99)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
$\sigma^2$  is a hyper-parameter which controls the regularization strength.

Data loss is the difference between the reconstruction picture and the orginal data picture. It's the data-fitting
term. The smaller $\sigma^2$ is, the effect of data loss  on the final loss is more  important. As a result, the generated 
pictures are more similar to the original pictures and the generalization effect is compromised.

kl_divergence can be interpreted as the information gained by using the posterior is more important than the 
prior distribution. It's the regularization term. The larger  $\sigma^2$ is, the less possibility that there appears
over-fitting in training process.

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, z_dim=0,
        data_label=0, label_noise=0.0,
        discriminator_optimizer=dict(
            type='SGD',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='SGD',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======

    hypers['batch_size'] = 32
    hypers['z_dim'] = 32
    hypers['data_label'] = 1
    hypers['label_noise'] = 0.1
    hypers['discriminator_optimizer'] = {'type': 'Adam', 'lr': 0.1 }
    hypers['generator_optimizer'] = {'type': 'Adam', 'lr': 0.1 }
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
The objective of the discriminator is to learn from this supplied dataset how to distinguish real 
from fake signals. During this part of GAN training, only the discriminator parameters are updated. 

When the fake signal is presented to the discriminator, naturally it will be classified
as fake with a label close to 0.0. The optimizer computes generator parameter updates 
based on the presented label (that is, 1.0) and its own prediction to take into account this 
new training data. In other words, the discriminator has some level of doubts about its 
prediction and GAN takes that into consideration. This time, GAN will let the gradients 
back propagate from the last layer of the discriminator down to the first layer of the 
generator. However, in most practices, during this phase of training, the discriminator 
parameters are temporarily frozen. The generator uses the gradients to update its parameters and 
improve its ability to synthesize fake signals.

"""

part3_q2 = r"""
**Your answer:**


1.No, we shouldn't stop training solely based on the fact that the Generator loss is below the threshold.
When the generator loss is small, the generated fake pictures is more similar to the real pictures and thus
more likely to confuse the discriminator. However, the ultimate goal of GAN is to reduce discriminator loss while
decrease the generator loss at the same time in order to get a discriminator which distinguishes the fake pictures 
which are very similar to the real pictures

2.The discriminator got too strong relative to the generator. Beyond this point, the generator finds it almost
impossible to fool the discriminator, hence the increase in it's loss.


"""

part3_q3 = r"""
**Your answer:**

The GAN generates better pictures than VAE's blurry output.
However, VAE learns hidden representation of data better.

Autoencoders learn a given distribution comparing its input to its output, this is good for learning hidden 
representations of data, but is pretty bad for generating new data. Mainly because we learn an averaged 
representation of the data thus the output becomes pretty blurry.

Generative Adversarial Networks take an entirely different approach. They use another network (so-called Discriminator) 
to measure the distance between the generated and the real data. Basically what it does is distinguishing 
the real data from the generated. It receives some data as an input and returns a number between 0 and 1. 
0 meaning the data is fake and 1 meaning it is real. The generators goal then is learning to convince 
the Discriminator into believing it is generating real data.



"""

# ==============


