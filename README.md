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

