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
    hypers.update(batch_size = 32, h_dim=1024, z_dim=512, x_sigma2=1., learn_rate=1e-4, betas=(0.9, 0.999),)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
        generator_optimizer=dict(
            type='',  # Any name in nn.optim like SGD, Adam
            lr=0.0,
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


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

# ==============


