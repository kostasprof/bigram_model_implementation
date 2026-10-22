# Character-Level Bigram Language Model

This project implements and compares two character-level bigram language models that generate names: a **statistical model** built from transition counts, and a **neural model** trained with PyTorch.

It covers conditional probabilities, Laplace smoothing, one-hot encoding, cross-entropy training, constrained sampling, and held-out evaluation with negative log-likelihood (NLL).

## Table of Contents

- [Overview](#overview)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Notation](#notation)
- [1. Statistical Bigram Model](#1-statistical-bigram-model)
- [2. Neural Bigram Model](#2-neural-bigram-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Example Generated Names](#example-generated-names)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Limitations and Possible Improvements](#limitations-and-possible-improvements)
- [Key Takeaways](#key-takeaways)

## Overview

A bigram language model predicts the next character from the current character alone. By learning which character pairs are common in names, it can generate new sequences that resemble the training data.

- **Statistical model:** estimates transition probabilities from counts and applies Laplace smoothing so unseen transitions keep a small non-zero probability.
- **Neural model:** learns a $28 \times 28$ weight matrix with gradient descent on a cross-entropy loss with an L2 penalty.

Both models are trained on the same training names and evaluated on the same held-out test names.

## Dataset and Preprocessing

The project uses `names.txt` from Andrej Karpathy's [makemore repository](https://github.com/karpathy/makemore/blob/master/names.txt), which contains 32,033 entries.

1. Load the names and strip whitespace.
2. Remove duplicates.
3. Split the unique names 80/20 into training and test sets, using `random_state=42` for reproducibility.

| Dataset | Number of names |
|---|---:|
| Training set | 23,595 |
| Test set | 5,899 |
| **Total unique names** | **29,494** |

## Notation

| Symbol | Meaning |
|---|---|
| $\langle s \rangle$ | start token (`<start>`) |
| $\langle e \rangle$ | end token (`<end>`) |
| $V$ | vocabulary: 26 lowercase letters, $\langle s \rangle$ and $\langle e \rangle$ (28 tokens) |
| $C(a,b)$ | number of times $b$ directly follows $a$ in the training data |
| $C(a)$ | number of transitions leaving $a$, i.e. $\sum_{b} C(a,b)$ |
| $\mathcal{T}$ | the set of transitions being evaluated, with $N = \lvert\mathcal{T}\rvert$ |

### Valid transitions

Two kinds of transition are impossible in this setup:

- No token can transition **into** $\langle s \rangle$.
- $\langle s \rangle \to \langle e \rangle$ is forbidden, because empty names are not allowed.

Nothing leaves $\langle e \rangle$, since a name ends there.

How each model handles these restrictions differs, and the difference matters when comparing their scores (see [Evaluation](#evaluation)):

| Stage | Statistical model | Neural model |
|---|---|---|
| Probability table / training | invalid transitions excluded | **not** excluded (softmax over all 28 tokens) |
| Evaluation | invalid transitions excluded | **not** excluded (softmax over all 28 tokens) |
| Generation | invalid transitions excluded | invalid transitions masked |

## 1. Statistical Bigram Model

### Bigram construction

Each name is padded with the start and end tokens. For example, `anna` produces:

```text
<start> -> a
a       -> n
n       -> n
n       -> a
a       -> <end>
```

A single-letter name such as `a` produces `<start> -> a` and `a -> <end>`. A name with $L$ letters always produces $L + 1$ transitions.

### Maximum-likelihood estimate

Without smoothing, the conditional probability of $b$ following $a$ is

$$
P(b \mid a) = \frac{C(a,b)}{C(a)}
$$

### Laplace smoothing

Some valid pairs never appear in the training data, and the unsmoothed estimate would give them probability zero. That would make the test NLL infinite whenever such a pair occurs in a test name. Laplace (add-one) smoothing adds one pseudo-count to every valid transition:

$$
P(b \mid a) = \frac{C(a,b) + 1}{C(a) + K_a}
$$

where $K_a$ is the number of valid next tokens after $a$:

$$
K_a =
\begin{cases}
26 & \text{if } a = \langle s \rangle \quad \text{(26 letters; } \langle e \rangle \text{ is not allowed)} \\
27 & \text{if } a \text{ is a letter} \quad \text{(26 letters and } \langle e \rangle \text{)}
\end{cases}
$$

Using $K_a$ in the denominator guarantees that $\sum_{b} P(b \mid a) = 1$ over the valid next tokens of each $a$. The function `check_probabilities` verifies this for every row.

### Name generation

Generation starts at $\langle s \rangle$. The next character is sampled from $P(\cdot \mid a)$ for the current character $a$, and this repeats until $\langle e \rangle$ is drawn.

## 2. Neural Bigram Model

### Vocabulary and encoding

Each of the 28 tokens is mapped to an integer (`a`-`z` are 0-25, `<start>` is 26, `<end>` is 27). The current character $a$ is converted to a one-hot vector $x_a \in \{0,1\}^{28}$.

### Architecture

The model is a single trainable matrix $W \in \mathbb{R}^{28 \times 28}$, initialised from a standard normal distribution. The logits for the next token are

$$
z = x_a^{\top} W
$$

which is simply row $a$ of $W$. During training and evaluation these are converted to probabilities with a softmax over **all 28 tokens**:

$$
P_W(b \mid a) = \frac{\exp(z_b)}{\sum_{j \in V} \exp(z_j)}
$$

### Training

The objective is the average cross-entropy over the training transitions plus an L2 penalty on the weights:

$$
L(W) = -\frac{1}{N} \sum_{(a,b) \in \mathcal{T}} \ln P_W(b \mid a) \; + \; \lambda \cdot \frac{1}{28^2} \sum_{i,j} W_{ij}^2
$$

The first term is the training NLL. The second is `mean(W**2)` scaled by $\lambda = 0.01$.

Gradients are computed with PyTorch autograd and the weights are updated manually with plain gradient descent:

$$
W \leftarrow W - \eta \, \nabla_W L(W)
$$

Training uses learning rate $\eta = 50$ for 1,000 epochs, with every training transition processed in one full batch. The training log prints $L(W)$ (the **regularised** objective, including the penalty) every 10 epochs.

### Name generation

Generation starts at $\langle s \rangle$. At each step the logits of impossible destinations are set to $-\infty$ before the softmax, so the remaining probabilities are renormalised over the valid next tokens only:

$$
P_{\text{gen}}(b \mid a) =
\begin{cases}
\dfrac{\exp(z_b)}{\sum_{j \,:\, (a,j) \text{ valid}} \exp(z_j)} & \text{if } (a,b) \text{ is valid} \\[2ex]
0 & \text{otherwise}
\end{cases}
$$

The next token is then sampled from this distribution, and generation stops when $\langle e \rangle$ is drawn. The decoded name excludes the special tokens.

## Evaluation

Both models are scored with the average negative log-likelihood per transition on the held-out test names:

$$
\mathrm{NLL} = -\frac{1}{N} \sum_{(a,b) \in \mathcal{T}} \ln P(b \mid a)
$$

where $\mathcal{T}$ contains every transition of every test name, **including** the $\langle s \rangle \to$ first-letter and last-letter $\to \langle e \rangle$ transitions. For a name with $L$ letters this adds $L + 1$ terms. The statistical model computes this with `negative_log_likelihood`, and the neural model with `evaluate`, which is the cross-entropy of the learned rows on the same test transitions.

A lower NLL means the model assigns higher probability to the observed transitions. As a reference point, guessing uniformly among 27 tokens would score $\ln 27 \approx 3.30$.

> **Compare models using test NLL, not the training objective.** The training objective includes the L2 penalty, so it is larger than the pure training NLL.

> **Caveat: the two scores are not perfectly like-for-like.** The statistical model's probabilities only cover valid transitions, but the neural model's softmax spreads probability over all 28 outputs during training and evaluation, including transitions that can never occur. Only generation applies the validity mask. Masking the logits before the cross-entropy in training and evaluation would remove this difference (see [Possible Improvements](#limitations-and-possible-improvements)).

## Results

The following results come from one execution of the program, with $\lambda = 0.01$.

### Model comparison

| Model | Dataset | Average NLL |
|---|---|---:|
| Statistical (Laplace) | Test | **2.4530** |
| Neural ($\lambda = 0.01$) | Test | 2.4587 |
| Neural ($\lambda = 0.01$) | Training | 2.4614 |

The statistical model had a slightly lower test NLL, by about 0.006. Because of the evaluation caveat above, and because these come from a single run, this gap should not be over-interpreted. The neural model has only 784 parameters and is trained on about 23,000 names, so there is very little overfitting for regularisation to prevent. It converges towards the same transition table the counts give directly, and the L2 penalty only pulls its probabilities slightly towards uniform.

### Effect of the regularisation strength

Changing $\lambda$ in `train()` and rerunning gave:

| $\lambda$ | Neural test NLL | Neural train NLL |
|---|---:|---:|
| 0.001 | 2.4539 | 2.4564 |
| 0.01 | 2.4587 | 2.4614 |

The weaker penalty was closer to the statistical model's 2.4530. With this much data, a larger $\lambda$ mainly adds bias.

### Neural training progress

Regularised objective $L(W)$ for the $\lambda = 0.01$ run:

| Epoch | Objective |
|---|---:|
| 0 | 3.7629 |
| 100 | 2.4937 |
| 500 | 2.4823 |
| 990 | 2.4819 |

The objective falls quickly in the first few dozen epochs and then flattens, indicating convergence. These values include the L2 penalty (about 0.02 at this $\lambda$) and are not the same as the training NLL in the table above.

## Example Generated Names

Statistical model (one execution):

```text
a
izyaa
taryshaniahe
faaha
mi
anan
tan
anirodayami
aahin
een
```

Neural model (one execution):

```text
nidesaxeva
```

Output differs between executions because characters are sampled randomly. A bigram model only sees the previous character, so it captures local letter patterns (common endings, plausible pairs) but not whole-name structure. That is why some outputs look like real names and others do not.

## Technologies Used

- **Python**: implementation and data processing.
- **NumPy**: sampling for the statistical model.
- **scikit-learn**: reproducible train/test split.
- **PyTorch**: tensors, automatic differentiation and the neural model.

## Project Structure

```text
bigram_model_implementation/
├── data/
│   └── names.txt
├── src/
│   ├── bigram.py
│   └── neural_bigram.py
└── README.md
```

- `bigram.py`: data loading, bigram counting, Laplace-smoothed probabilities, name generation, NLL evaluation, and the `main()` entry point that also runs the neural model.
- `neural_bigram.py`: integer encoding, neural training, name generation, evaluation and decoding.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/kostasprof/bigram_model_implementation.git
cd bigram_model_implementation
```

### 2. Install the dependencies

```bash
python -m pip install numpy scikit-learn torch
```

### 3. Prepare the dataset

Download `names.txt` from the [original source](https://github.com/karpathy/makemore/blob/master/names.txt) and place it at:

```text
data/names.txt
```

The script finds it relative to its own location, so it works from any working directory:

```python
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "names.txt"
```

### 4. Run

From the repository root:

```bash
python src/bigram.py
```

The program will:

1. Load, deduplicate and split the dataset.
2. Build the bigram counts and Laplace-smoothed probabilities.
3. Generate 10 example names with the statistical model.
4. Print the statistical model's test NLL.
5. Train the neural model, printing its regularised objective every 10 epochs.
6. Print the statistical test NLL, the neural test NLL and the neural training NLL.
7. Generate one name with the trained neural model.

## Limitations and Possible Improvements

- **Single-character context.** Each prediction depends only on the previous character, so longer-range structure cannot be captured.
- **Mask not applied in neural training and evaluation.** Only `generate()` blocks impossible transitions. Setting those logits to $-\infty$ before the cross-entropy in `train()` and `evaluate()` would make the neural model's NLL directly comparable to the statistical model's.
- **L2 is not Laplace smoothing.** $\lambda$ is hardcoded to 0.01. Try $\lambda = 0$ and a small sweep; with this much data the best neural result should sit very close to the statistical model.
- **Single run.** There are no random seeds and results come from one initialisation and one split. Averaging over several seeds would show how much of the gap between models is noise.
- **Metrics.** Report perplexity, $\exp(\mathrm{NLL})$, alongside NLL.
- **Optimisation.** Tune the learning rate and epoch count, or use mini-batches and an optimiser such as Adam.
- **Richer models.** Extend to trigrams or a neural model with a longer context window.

## Key Takeaways

Counting and gradient descent arrive at nearly the same bigram distribution: the neural model learns by optimisation the transition table that the statistical model computes directly. A simple, well-smoothed baseline is a strong reference point, and a more complex model has to beat it on held-out data to justify its complexity. The comparison also shows why evaluation needs to be consistent: the two models should be scored over the same set of valid transitions before drawing conclusions from small differences in NLL.
