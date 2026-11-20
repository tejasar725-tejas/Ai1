# 🍳 Recipe Step Autocompleter

> A Command Line Interface (CLI) tool that helps home cooks autocomplete incomplete recipe instructions, ensuring smooth cooking and baking experiences.

[![CI](https://github.com/example/recipe-step-autocompleter/actions/workflows/ci.yml/badge.svg)](https://github.com/example/recipe-step-autocompleter)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

Cooking often involves multi-step processes where home cooks forget the exact next action or measurement (e.g., *"Heat olive oil in a..."*, *"Preheat oven to..."*, or *"Whisk together the flour,..."*). 

**Recipe Step Autocompleter** compiles a domain-specific corpus of culinary instructions, calculates n-gram statistics with Maximum-Likelihood Estimation (MLE), applies smoothing techniques (Laplace, Lidstone, and Backoff), and generates natural, coherent continuations.

The tool includes:
- A statistical **N-Gram Language Model** ($n=2, 3, 4$) with Laplace (Add-1), Lidstone (Add-k), and Backoff smoothing.
- Robust culinary **text preprocessing and tokenization** (handling vulgar fractions like `½`, temperature ranges, units, and contractions).
- Controllable **decoding strategies** (deterministic greedy, temperature sampling, top-k candidate filtering, and repetition/cycle suppression).
- A feature-rich **CLI** supporting single completions, multi-phrase candidate generation, interactive shell mode (`Chef >`), corpus statistics, and model perplexity evaluation.
- Optional **LLM hybrid integration** supporting Gemini, Groq, Ollama, or mock fallbacks via `.env`.

---

## 📂 Project Structure

```
recipe-step-autocompleter/
├── .env.example              # Template configuration for LLM providers & defaults
├── .env                      # Active environment configuration
├── .gitignore                # Git ignore rules for Python & environments
├── .github/
│   └── workflows/
│       └── ci.yml            # CI test runner matrix (Python 3.12 & 3.13)
├── README.md                 # Complete documentation & usage guide
├── pytest.ini                # Pytest configuration and test options
├── requirements.txt          # Python dependencies
├── data/
│   └── recipes.txt           # Curated corpus of authentic cooking steps
├── app/
│   ├── __init__.py           # Package exports and version metadata
│   ├── cli.py                # Command Line Interface (hello, autocomplete, interactive, etc.)
│   ├── core.py               # Preprocessing, tokenization, N-Gram model, MLE & smoothing
│   ├── llm.py                # Optional LLM providers (Gemini, Groq, Ollama, Mock)
│   └── storage.py            # Corpus loading, file management, model serialization
└── tests/
    ├── __init__.py
    ├── test_smoke.py         # Smoke tests (CLI hello, basic imports, fallback sanity)
    └── test_core.py          # 30+ unit tests covering all milestones & stretch goals
```

---

## 🎯 Milestones & Capabilities

- [x] **Milestone 1: Compile a corpus of cooking recipe instructions**
  - Text normalization: converts unicode fractions (`½` $\to$ `1/2`), standardizes quotes/dashes, trims whitespace, handles casing.
  - Culinary tokenization: preserves fractions (`1/2`), contractions (`don't`), measurements, and separates punctuation.
  - Sentence padding (`<s>`, `</s>`) and out-of-vocabulary (`<unk>`) token mapping.
  - Detokenization: formats punctuation spacing and capitalizes sentences.
- [x] **Milestone 2: Implement N-gram counting and maximum-likelihood probabilities with smoothing**
  - Flexible n-gram counter supporting arbitrary orders ($n=1, 2, 3, 4$).
  - Maximum-Likelihood Estimation ($P_{MLE} = C(c, w) / C(c)$).
  - Laplace (Add-1) smoothing ($P_{Laplace} = \frac{C(c, w) + 1}{C(c) + |V|}$).
  - Lidstone (Add-k) smoothing ($P_{Lidstone} = \frac{C(c, w) + k}{C(c) + k|V|}$).
  - Stupid Backoff smoothing backing down from trigram $\to$ bigram $\to$ unigram with discount $\alpha=0.4$.
- [x] **Milestone 3: Generate recipe step continuations**
  - Next-word distribution calculation and top-k candidate ranking.
  - Greedy decoding (deterministic argmax) and stochastic temperature sampling.
  - Repetition and cycle suppression to prevent degenerate loops.
  - Termination criteria on end-of-step token `</s>` or terminal punctuation (`.`, `!`).
- [x] **Milestone 4: Build a functional CLI for user interaction**
  - `hello` command: greets user and prints model diagnostics.
  - `autocomplete` command: accepts partial recipe steps and generates continuations.
  - `interactive` command: interactive REPL session (`Chef >`).
  - `stats` command: displays corpus stats and perplexity metrics.
  - `add` command: adds new recipe instructions to the active corpus.
- [x] **Stretch Ideas**
  - **Multiple suggestions**: Generate $k$ distinct alternative continuations (`-k 3`).
  - **Customizable length**: Set suggestion length in words (`-l 12`).
  - **Smoothing exploration**: Switch smoothing methods on the fly (`-s laplace|lidstone|backoff|mle`).
  - **Perplexity evaluation**: Quantitatively assess model quality across smoothing methods.

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Requires **Python 3.12+** and `git`.

```bash
# Clone the repository
git clone https://github.com/example/recipe-step-autocompleter.git
cd recipe-step-autocompleter

# Create virtual environment and activate
python -m venv .venv
source .venv/bin/activate      # On Linux/macOS
# or .\.venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example configuration file:

```bash
cp .env.example .env
```

To enable optional LLM hybrid completion (e.g., Gemini, Groq, or Ollama), edit `.env` and set `LLM_PROVIDER` and your API key. If left as `none`, the tool runs purely on the statistical n-gram model.

### 3. Verify Installation

Run the greeting command:

```bash
python -m app.cli hello
```

Expected output:
```
=================================================================
 🍳 Recipe Step Autocompleter CLI - Welcome Home Chef! 🍳
=================================================================
Status       : Ready to help home cooks autocomplete recipe steps.
Corpus Steps : 86 authentic recipe instructions loaded.
Default Model: 3-Gram Statistical Language Model
Smoothing    : Laplace (Add-1) + Lidstone + Backoff supported
LLM Provider : none (Active: False)
-----------------------------------------------------------------
Quick Start Examples:
  python -m app.cli autocomplete "Heat olive oil in a"
  python -m app.cli autocomplete "Preheat oven to" --length 12 --num-suggestions 3
  python -m app.cli autocomplete "Whisk together flour" --method lidstone
  python -m app.cli interactive
=================================================================
```

---

## 💻 CLI Usage Guide

### Autocomplete a Partial Recipe Step

```bash
python -m app.cli autocomplete "Heat olive oil in a"
```

Output:
```
=================================================================
Input Partial Step: "Heat olive oil in a"
Model: 3-gram | Smoothing: laplace | Length: 10 words
=================================================================

Generated Continuation:
  👉 Large pot of salted water to a gentle simmer.

Complete Recipe Step:
  🍳 Heat olive oil in a large pot of salted water to a gentle simmer.
=================================================================
```

### Suggest Multiple Alternatives (Stretch Goal 1)

Generate 3 distinct suggestions with `-k 3`:

```bash
python -m app.cli autocomplete "Preheat oven to" --length 12 -k 3
```

Output:
```
Top 3 Suggestions:

  [1] Continuation: 375 degrees f (190 degrees c) and grease a 9x13-inch
      Full Step   : Preheat oven to 375 degrees f (190 degrees c) and grease a 9x13-inch

  [2] Continuation: 425 degrees f (220 degrees c) and line a large
      Full Step   : Preheat oven to 425 degrees f (220 degrees c) and line a large

  [3] Continuation: 400 degrees f.
      Full Step   : Preheat oven to 400 degrees f.
```

### Specify Suggestion Length (Stretch Goal 2)

Control the continuation length with `-l <words>`:

```bash
# Short 4-word continuation
python -m app.cli autocomplete "Whisk together the" -l 4

# Longer 15-word continuation
python -m app.cli autocomplete "Whisk together the" -l 15
```

### Select Smoothing Method (Stretch Goal 3)

Compare different smoothing techniques using `-s`:

```bash
# Laplace (Add-1) smoothing
python -m app.cli autocomplete "Melt butter in a" -s laplace

# Lidstone (Add-k) smoothing
python -m app.cli autocomplete "Melt butter in a" -s lidstone

# Stupid Backoff smoothing
python -m app.cli autocomplete "Melt butter in a" -s backoff
```

### Interactive Chef Mode

Launch the interactive REPL where you can type prompts interactively:

```bash
python -m app.cli interactive
```

```text
Chef > Bring a large pot of
--------------------------------------------------
  [1] ... Salted water to a rolling boil over high heat.
      Full: Bring a large pot of salted water to a rolling boil over high heat.
--------------------------------------------------
Chef > :n 4
Switched to 4-gram model.
Chef > exit
Goodbye and happy cooking!
```

### View Corpus Statistics & Perplexity

Inspect token counts, vocabulary size, and perplexity across smoothing methods:

```bash
python -m app.cli stats
```

Output:
```
=================================================================
 📊 Recipe Corpus & N-Gram Model Statistics 📊
=================================================================
Recipe Steps in Corpus : 86
Total Tokens Trained   : 1546
Vocabulary Size (|V|)  : 485
Configured N-Gram Order: 3
  Order 1: 1 distinct contexts, 1804 n-gram occurrences
  Order 2: 483 distinct contexts, 1718 n-gram occurrences
  Order 3: 1112 distinct contexts, 1632 n-gram occurrences

Perplexity Evaluation (lower is better):
  Smoothing: laplace    => Perplexity: 260.81
  Smoothing: lidstone   => Perplexity: 202.70
  Smoothing: backoff    => Perplexity: 13.35
=================================================================
```

### Add New Recipe Instructions to Corpus

```bash
python -m app.cli add "Simmer marinara sauce over low heat for 30 minutes."
```

---

## 📐 Mathematical Formulation

### 1. Maximum-Likelihood Estimation (MLE)
Given context $c = (w_{i-n+1}, \dots, w_{i-1})$:
$$P_{MLE}(w_i \mid c) = \frac{C(c, w_i)}{C(c)}$$

### 2. Laplace (Add-1) Smoothing
Adds pseudo-count of 1 to every word in vocabulary $V$:
$$P_{Laplace}(w_i \mid c) = \frac{C(c, w_i) + 1}{C(c) + |V|}$$

### 3. Lidstone (Add-k) Smoothing
Adds pseudo-count $k \in (0, 1]$ (default $k=0.5$):
$$P_{Lidstone}(w_i \mid c) = \frac{C(c, w_i) + k}{C(c) + k|V|}$$

### 4. Stupid Backoff
Recursively backs off to $(n-1)$-grams discounted by factor $\alpha = 0.4$:
$$S(w_i \mid w_{i-k+1}^{i-1}) = \begin{cases} \frac{C(w_{i-k+1}^i)}{C(w_{i-k+1}^{i-1})} & \text{if } C(w_{i-k+1}^i) > 0 \\ \alpha \cdot S(w_i \mid w_{i-k+2}^{i-1}) & \text{otherwise} \end{cases}$$

### 5. Perplexity (Evaluation Metric)
Given a test sequence of $N$ tokens:
$$PP(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid c_i) \right)$$

---

## 🧪 Running Tests

Run the full automated test suite with pytest:

```bash
pytest
```

Run with verbose details:
```bash
pytest -v
```

Run only smoke tests:
```bash
pytest tests/test_smoke.py
```

All 37 unit and integration tests pass cleanly across platforms and Python 3.12/3.13.

---

## 🤝 Contributing & License

Contributions, extra recipe corpora, and new smoothing techniques are welcome!
Licensed under the [MIT License](LICENSE).
