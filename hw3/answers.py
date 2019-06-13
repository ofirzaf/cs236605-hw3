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
    temperature = 0.4
    start_seq = "ACT I."
    # ========================
    return start_seq, temperature


part1_q1 = r"""
We split the corpus to sequences because we want to speed up the training process by training in batches.
Furthermore in longer sequences the gradients will vanish and there is no benefit from that.
Also in terms of memory in most cases we wont be able to load the whole corpus into the GPU's memory 
"""

part1_q2 = r"""
The generated text shows memory longer than the sequence length because we pass the hidden state between batches. 
The hidden state can be viewed as the accumulated memory of the sequence or as the context of the next character in 
the sequence. 
"""

part1_q3 = r"""
We are not shuffling the dataset because we want to keep the hidden state between batches in order to the model to 
be able to generate longer sequences.
If we would have shuffled the batches there won't be any connection between one batch and the other.
"""

part1_q4 = r"""
1. When generating sequences we want the model to predict the characters 
with highest probabilities, according to our model, in order to generate sequences that have real meaning. 
In order to achieve that we lower the temperature in order to sample only the highest probability characters.
2. When the temperature is too high we make the distribution of sampling move towards uniform distribution according to 
the graph we produced in the notebook. This will result in randomly sampled characters and a sequence without meaning.
3. When the temperature is very low we give more weight to the model's predictions and this will result in sampling 
only the highest probability characters. The sampling distribution will move towards delta distribution.
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
    hypers.update(batch_size=8, h_dim=1024, z_dim=20, x_sigma2=0.0001, learn_rate=1e-5, betas=(0.9, 0.997),)
    # ========================
    return hypers


part2_q1 = r"""
The $\sigma^2$ hyperparameter defines how much we want to allow the output diverge from the real input image.<br>
For very high $\sigma^2$ we will get very random images because the data loss will be approximately 0. However the latent distribution will be very similar to normal distribution since the loss will be dominated by the KL divergence.<br>
For very low $\sigma^2$ we will make the decoder learn for every point the latent space a single image from the dataset since the loss will be dominated by the data loss expression. This will also result in an unexpected latent space distribution since the KL divergence will not affect the loss at all.
"""

part2_q2 = r"""
1. The reconstruction part of the loss role is to make sure that the decoded images will be similar to the training data in order to make sure that we sample images from the same data space. The KL divergence part purpose is to make sure that the approximated latent space distribution is similar to the true latent space distribution which is a normal distribution.
2. The KL divergence term makes the latent space distribution similar to the normal distribution from which we sample $z$
3. The latent space distribution becomes more dense and without gaps so when we sample it (from a normal distribution) the decoder can effectively decode.
Moreover since we want to generate images by sampling from the latent space we need to know it's distribution in order to get good sampling results.
"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=4, z_dim=32,
        data_label=1, label_noise=0.3,
        discriminator_optimizer=dict(
            type='SGD',  # Any name in nn.optim like SGD, Adam
            lr=1000e-5,
        ),
        generator_optimizer=dict(
            type='Adam',  # Any name in nn.optim like SGD, Adam
            lr=100e-5,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # ========================
    return hypers


part3_q1 = r"""
We are training 2 different networks concurently which have adversary objectives.<br>
The discriminator parameters gradients are not affected at all by the generator's parameter's gradients, moreover the gradients of the generator's parameters calculated w.r.t the discriminator loss will tend to make the generator worse because the objective is to minimize the discriminator loss.<br>
When training the generator we calculate gradients in both networks since the gradients of the generator's parameters are calculated w.r.t the discriminator's gradients.
"""

part3_q2 = r"""
1. No, since we are training the networks on adverse objectives, in every step both networks will get better at their objectives. If the genertor loss is below some threshold is doesn't mean that it converged since the loss function is also dependent on the discriminator.
2. It means that the generator learns faster than the discriminator, in order to improve learning it is better to enhance the discriminator's learning.
"""

part3_q3 = r"""

"""

# ==============


