# Character-Level Bigram Language Model

This project implements and compares two character-level bigram language models for generating names: a statistical model based on character frequencies and a neural model implemented with PyTorch.

The project explores conditional probabilities, Laplace smoothing, neural network training, stochastic text generation, and model evaluation using negative log-likelihood (NLL).

## Table of Contents

- [Overview](#overview)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Statistical Bigram Model](#1-statistical-bigram-model)
- [Neural Bigram Model](#2-neural-bigram-model)
- [Results](#results)
- [Example Generated Names](#example-generated-names)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Limitations and Possible Improvements](#limitations-and-possible-improvements)
- [Key Takeaways](#key-takeaways)

## Overview

A bigram language model predicts the next character based on the current character. By learning which character pairs frequently occur in names, the model can generate new character sequences that resemble the training data.

This project implements two approaches to the same task:

- **Statistical Bigram Model:** Calculates character-transition probabilities from observed frequencies and applies Laplace smoothing to handle unseen transitions.
- **Neural Bigram Model:** Learns character-transition probabilities using a trainable weight matrix, one-hot encoded inputs, cross-entropy loss, and gradient-based optimisation.

Both models use the same training and test datasets, allowing their performance to be examined using negative log-likelihood.

## Dataset and Preprocessing

The project uses the `names.txt` dataset from Andrej Karpathy's [makemore repository](https://github.com/karpathy/makemore/blob/master/names.txt).

The original file contains **32,033 entries**. The following preprocessing steps are applied:

1. Load the names from the text file.
2. Remove duplicate names.
3. Split the unique names into training and testing sets using an 80/20 split.
4. Use `random_state=42` to make the split reproducible.

After removing duplicates, the dataset contains **29,494 unique names**.

| Dataset | Number of names |
|---|---:|
| Training set | 23,595 |
| Test set | 5,899 |
| Total unique names | 29,494 |

The training set is used to learn the character-transition probabilities and train the neural model. The held-out test set is used to evaluate both approaches.

## 1. Statistical Bigram Model

The statistical model estimates transition probabilities directly from character-pair frequencies in the training dataset.

### Bigram construction

Each name is represented using special start and end tokens.

For example, the name `anna` is represented by the following transitions:

```text
<start> → a
a       → n
n       → n
n       → a
a       → <end>
```

The model counts how frequently each bigram appears in the training data.

### Conditional probabilities

The probability of transitioning from character \(a\) to character \(b\) is estimated using:

\[
P(b\mid a)=\frac{C(a,b)}{C(a)}
\]

where:

- \(C(a,b)\) is the number of occurrences of the bigram \((a,b)\).
- \(C(a)\) is the total number of transitions originating from character \(a\).

### Laplace smoothing

Some character pairs may not appear in the training dataset. Assigning these pairs a probability of zero would prevent the model from generating them.

To address this, the model uses Laplace smoothing:

\[
P(b\mid a)=\frac{C(a,b)+1}{C(a)+K}
\]

Here, \(K\) is the number of valid possible next tokens for the current character. The implementation uses 27 possible destinations for ordinary letters and 26 for the start token, since an empty name is not allowed.

Only valid transitions are included in the probability distributions.

### Name generation

Name generation begins at `<start>`. The next character is sampled according to the learned transition probabilities, and the process repeats until `<end>` is selected.

Because generation is probabilistic, the model can produce familiar-looking names as well as unusual character combinations.

### Evaluation using negative log-likelihood

The statistical model is evaluated on the held-out test set using average negative log-likelihood:

\[
\operatorname{NLL}
=-\frac{1}{N}\sum_{i=1}^{N}\ln P(x_i)
\]

where \(N\) is the total number of character transitions evaluated, including the start and end transitions for each name.

A lower NLL indicates that the model assigns higher average probability to the observed transitions.

## 2. Neural Bigram Model

The second approach learns character-transition probabilities through training with PyTorch rather than estimating them directly from observed frequencies.

### Vocabulary and encoding

The model uses a vocabulary of 28 tokens:

- 26 lowercase English letters.
- One start token, `<start>`.
- One end token, `<end>`.

Each token is mapped to an integer. The input characters are converted into one-hot vectors, allowing the model to represent each current character as a vector of length 28.

### Model architecture

The model uses a trainable \(28\times28\) weight matrix.

The input one-hot vector is multiplied by the weight matrix to produce logits representing the possible next characters. The logits are converted into a probability distribution using softmax.

The architecture is intentionally simple, providing an introduction to learning character-transition probabilities with a neural model.

### Training

The model is trained on the same character transitions used by the statistical model.

The training process includes:

- Cross-entropy loss for next-character prediction.
- L2 regularisation to penalise large weights.
- Gradient calculation through PyTorch autograd.
- Manual gradient-based weight updates.
- 1,000 training epochs.

The training objective combines cross-entropy loss with an L2 regularisation term:

\[
L=L_{\mathrm{CE}}+\lambda\operatorname{mean}(W^2)
\]

where \(W\) is the weight matrix and \(\lambda=0.01\).

### Name generation

To generate a name, the neural model begins at the start token and repeatedly samples the next token from the softmax distribution.

The generation procedure masks invalid transitions, including transitions into `<start>` and the empty-name transition from `<start>` to `<end>`. Generation stops when the end token is selected.

### Evaluation

The neural model's average cross-entropy is used as its NLL score on the held-out test transitions. Its training and test scores can then be compared with the statistical model's NLL.

## Results

The following results were obtained from one execution of the program.

### Model comparison

| Model | Dataset | Average NLL |
|---|---|---:|
| Statistical Bigram Model | Test | **2.4530** |
| Neural Bigram Model | Test | 2.4587 |
| Neural Bigram Model | Training | 2.4614 |

The statistical model achieved a test NLL of approximately **2.4530**, while the neural model achieved **2.4587**.

In this run, the statistical model achieved a slightly lower test NLL than the neural model.

This demonstrates that a neural approach does not automatically outperform a simpler statistical model. Model complexity alone does not guarantee better predictive performance.

### Neural training progress

The neural model's printed training objective decreased during training:

| Epoch | Training objective |
|---|---:|
| 0 | 3.7933 |
| 100 | 2.4935 |
| 500 | 2.4822 |
| 990 | 2.4819 |

The objective decreased substantially during the initial epochs and then stabilised.

These values represent the **regularised training objective**, which includes both cross-entropy and the L2 penalty. They should not be interpreted as pure training NLL.

### Important evaluation note

The current neural implementation masks invalid transitions during generation but does not apply the same restrictions during training and evaluation. Consequently, the statistical and neural models' NLL scores are not a perfectly like-for-like comparison.

The reported results reflect the current implementation and should be interpreted with this limitation in mind.

## Example Generated Names

The following names were generated by the statistical model during one execution:

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

The neural model generated the following name:

```text
nidesaxeva
```

The outputs vary between executions because characters are sampled probabilistically. Some generated names resemble familiar names, while others contain unusual combinations of letters.

The goal is to learn character-level patterns from the dataset, not to guarantee that every generated name is an existing or correctly spelled name.

## Technologies Used

- **Python** — implementation and data processing.
- **NumPy** — probabilistic sampling for the statistical model.
- **scikit-learn** — reproducible training and test split.
- **PyTorch** — neural model, tensor operations, automatic differentiation, and optimisation.

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

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/kostasprof/bigram_model_implementation.git
cd bigram_model_implementation
```

### 2. Install the dependencies

Make sure Python and pip are installed, then run:

```bash
python -m pip install numpy scikit-learn torch
```

### 3. Prepare the dataset

Download `names.txt` from the [original dataset source](https://github.com/karpathy/makemore/blob/master/names.txt) and place it in the `data/` directory.

The project expects the following file:

```text
data/names.txt
```

### 4. Configure the dataset path

To make the project portable across computers, define the dataset path relative to the script rather than using an absolute path tied to one computer.

In `src/bigram.py`, use:

```python
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "names.txt"
```

### 5. Run the project

From the repository root, execute:

```bash
python src/bigram.py
```

The program will:

1. Load and split the dataset.
2. Generate example names using the statistical model.
3. Calculate the statistical model's test NLL.
4. Train the neural model and display its training progress.
5. Evaluate the neural model on the training and test transitions.
6. Generate a name using the trained neural model.

## Limitations and Possible Improvements

The project uses a simple bigram architecture, so each character prediction depends only on the immediately preceding character. The models cannot directly capture longer character sequences or broader linguistic context.

Potential improvements include:

- Applying identical valid-transition constraints during neural training, evaluation, and generation.
- Comparing perplexity alongside NLL.
- Experimenting with different learning rates and regularisation strengths.
- Comparing results across multiple neural initialisations.
- Exploring trigram models or neural architectures that use a longer character context.

## Key Takeaways

This project explores two approaches to character-level language modelling: probability estimation from observed frequencies and learning transition probabilities through neural network training.

It demonstrates bigram counting, conditional probability estimation, Laplace smoothing, stochastic text generation, one-hot encoding, cross-entropy optimisation, and held-out evaluation using negative log-likelihood.

The comparison also highlights the importance of using a simple baseline, evaluating models consistently, and interpreting results in the context of each implementation's limitations.
