# Character-Level Bigram Language Model

A from-scratch bigram language model that learns which letters tend to follow which other letters in a list of names, then uses those statistics to score and generate names. Probabilities are estimated by counting, with Laplace (add-one) smoothing so unseen letter pairs never get zero probability.

## How it works

1. **Load and split the data** (`handle_input`)
   Names are read from `data/names.txt`, stripped of newlines, de-duplicated, and split 80/20 into train and test sets (`random_state=42` for reproducibility).

2. **Extract bigrams** (`get_pairs`)
   Each name is wrapped with special tokens, so `ana` becomes the transitions `<start>→a`, `a→n`, `n→a`, `a→<end>`. Single-letter names produce `<start>→x` and `x→<end>`.

3. **Count bigrams** (`count_bigrams`)
   Counts how often each `(current, next)` pair appears in the training set.

4. **Compute probabilities with Laplace smoothing** (`calculate_probabilities`)
   For a pair `(a, b)`:

   ```
   P(b | a) = (count(a, b) + 1) / (count(a) + V)
   ```

   where `count(a)` is the total number of bigrams starting with `a`, and `V` is the number of valid next tokens:

   | Current token | Valid next tokens | `V` |
   |---|---|---|
   | a–z | a–z and `<end>` | 27 |
   | `<start>` | a–z only (an empty name `<start>→<end>` is not allowed) | 26 |
   | `<end>` | none (never a source) | n/a |

   Every pair that is valid but never seen in training is added with a count of 0, so it still receives a small non-zero probability.

5. **Generate names** (`generate_names`)
   Starting at `<start>`, repeatedly sample the next character from the learned distribution until `<end>` is drawn.

6. **Evaluate** (`negative_log_likelihood`)
   Reports the average negative log likelihood (NLL) per transition over the test set. Lower is better.

## Project structure

```
.
├── data/
│   └── names.txt        # one name per line
├── src/
│   └── bigram.py        # the model
└── README.md
```

## Requirements

- Python 3.8+
- `numpy`
- `scikit-learn`

```bash
pip install numpy scikit-learn
```

## Usage

Make sure `data/names.txt` exists (one lowercase name per line), then run:

```bash
python src/bigram.py
```

Run it from the project root so the relative `data/names.txt` path resolves. The script trains the model on the training split, prints the train/test sizes, generates 10 sample names, and prints the test-set NLL.

Example output:

```
Bigram Language Model
Train dataset names:23595 Test dataset names:5899
Generated name:pe
Generated name:seyan
Generated name:mele
...
The NLL score is: 2.4529519648377267
```

### Optional utilities

These functions are included but not called by default:

- **`check_probabilities(probs)`**: sanity check that the outgoing probabilities of each starting token sum to 1. Uncomment the call in `main()` to run it.
- **`evaluation(test, probs)`**: picks a random test name and returns its probability under the model. Uncomment the loop in `main()` to run it.
- **`generate_names(probs)`**: samples a new name from the model (called in `main()` to print sample names).

  ```python
  train, test = handle_input()
  probs = calculate_probabilities(count_bigrams(get_pairs(train)))
  print(generate_names(probs))
  ```

## Results

| Metric | Value |
|---|---|
| Dataset size (unique names) | 29,494 |
| Train / test split | 23,595 / 5,899 |
| Test NLL (avg. per transition) | 2.453 |

For reference, a model that picks uniformly among 27 possible next tokens would score an NLL of ln(27) ≈ 3.30, so the bigram model's 2.45 shows it has learned real structure from the data.

**Sample generated names:**

```
pe
aynndaniyalisirorieir
ll
z
seyan
tha
n
ane
mele
akewa
```

The samples show the typical bigram behavior: some outputs look like plausible names (`seyan`, `mele`, `akewa`), while others are too short (`z`, `n`, `ll`) or run on far too long (`aynndaniyalisirorieir`), because the model only knows the previous letter and has no sense of overall name length or structure.

## Assumptions and limitations

- **Lowercase a–z only.** The vocabulary is `a–z` plus `<start>` and `<end>`. A name containing uppercase letters, accents, hyphens, or spaces will raise a `KeyError` during evaluation, so clean or filter the data first.
- **One character of context.** A bigram model only sees the previous letter, so it can't capture longer patterns like common suffixes or syllables. Generated names are often name-like but not always pronounceable.
- **Training-set vocabulary.** Smoothing is applied over the fixed alphabet, not a vocabulary learned from the data.

## The Next Step

- Replacing the lookup tables with a small neural network trained on the same task

