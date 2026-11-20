https://ask.thili.ai/api/v1/public/projects/CODE/badge.svg)](https://ask.thili.ai/b/CODE
# 🍳 Recipe Step Autocompleter

A command-line tool that helps home cooks autocomplete incomplete recipe instructions, ensuring smooth cooking and baking experiences.

---

## 📖 Overview

Cooking often involves multi-step recipes where home cooks forget the exact next step, measurement, or cooking technique (e.g., *"Heat olive oil in a..."*, *"Preheat oven to..."*, or *"Whisk together the flour,..."*). 

**Recipe Step Autocompleter** addresses this by:
1. Compiling a domain-specific corpus of recipe instructions.
2. Building an **$N$-Gram Language Model** with Maximum-Likelihood Estimation (MLE) and smoothing (Laplace, Lidstone, and Backoff).
3. Generating natural, grammatically correct step continuations using greedy and temperature-scaled sampling.
4. Providing an intuitive **CLI** for real-time suggestions, multi-candidate completions, and interactive shell mode.

---

## 📂 Repository File Structure

The project is organized directly at the repository root into the `app/` package, `data/` directory, and `tests/` suite:

```
.
├── app/
│   ├── __init__.py           # Package exports and version metadata
│   ├── cli.py                # CLI commands: hello, autocomplete, interactive, stats, add
│   ├── core.py               # Preprocessing, tokenization, N-Gram model, MLE, smoothing & sampling
│   ├── llm.py                # Optional LLM integration (Gemini, Groq, Ollama, Mock fallback)
│   └── storage.py            # Corpus loading, dataset persistence, model artifact serialization
├── data/
│   └── recipes.txt           # Primary recipe instruction dataset (86 curated steps)
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py         # Smoke tests (CLI hello, basic imports, fallback sanity)
│   └── test_core.py          # 37 unit tests covering all milestones & stretch goals
├── .env.example              # Template configuration for model settings & optional API keys
├── .github/
│   └── workflows/
│       └── ci.yml            # Automated CI workflow on GitHub Actions
├── .gitignore                # Git ignore patterns for Python, venv, and cache
├── pytest.ini                # Pytest configuration and test options
├── requirements.txt          # Pinned dependency specifications
└── README.md                 # Complete documentation, data details, and run instructions
```

---

## 📊 Dataset & Corpus

The model trains on authentic culinary instructions stored at `data/recipes.txt`.

- **Location**: `data/recipes.txt`
- **Format**: UTF-8 plain text, one complete cooking instruction per line.
- **Corpus Size**: 86 curated recipe instructions containing 1,546 words across 485 unique vocabulary tokens.
- **Techniques Covered**:
  - *Sautéing & Pan-frying*: olive oil, butter, garlic, onions, mushrooms, searing proteins.
  - *Boiling & Pasta*: boiling salted water, pasta al dente, reserving pasta water.
  - *Baking & Roasting*: oven temperatures (350°F–425°F), flour, baking powder, folding batter.
  - *Soups, Stews & Sauces*: broth simmering, rouxs, heavy cream, reduction.
  - *Seasoning & Garnishing*: kosher salt, freshly cracked pepper, herbs, citrus.
  - *Dressings & Marinades*: vinaigrettes, whisking marinades, resting cooked meats.
- **Custom Corpora**: Users can provide their own recipe files using the `--corpus` flag:
  ```bash
  python -m app.cli autocomplete "Heat olive oil" --corpus path/to/my_recipes.txt
  ```
- **Fallback Guarantee**: If `data/recipes.txt` is absent, `app/storage.py` automatically restores the built-in corpus, ensuring zero-configuration execution.

---

## 📦 Pinned Dependencies

Dependencies are strictly pinned in `requirements.txt`:

| Package | Pinned Version | Purpose |
|---|---|---|
| `pytest` | `9.2.0` | Test runner and test suite execution |
| `python-dotenv` | `1.2.4` | Loads configuration from `.env` |
| `requests` | `2.34.2` | Optional HTTP client for external LLM endpoints |

Install them using:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

All commands are executed from the **repository root**:

### Step 1: Set Up Python Environment
Requires **Python 3.12** or **3.13**:
```bash
# On Linux / macOS:
python3.12 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 2: Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Copy Environment Configuration
```bash
# On Linux / macOS:
cp .env.example .env

# On Windows:
copy .env.example .env
```
*(By default, `LLM_PROVIDER=none`, meaning the tool runs purely offline on the statistical $N$-gram model. To enable optional LLM hybrid completion, see `.env.example`.)*

### Step 4: Run the CLI Greeting & Health Check
```bash
python -m app.cli hello
```
**Expected Output:**
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

### Step 5: Autocomplete a Recipe Step
```bash
python -m app.cli autocomplete "Heat olive oil in a"
```
**Actual Output:**
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

---

## 💡 Advanced CLI Features & Stretch Goals

### 1. Suggest Multiple Distinct Continuations (`-k`)
Generate multiple diverse alternative steps for the cook:
```bash
python -m app.cli autocomplete "Preheat oven to" --length 12 -k 3
```
**Actual Output:**
```
=================================================================
Input Partial Step: "Preheat oven to"
Model: 3-gram | Smoothing: laplace | Length: 12 words
=================================================================

Top 3 Suggestions:

  [1] Continuation: 375 degrees f (190 degrees c) and grease a 9x13-inch
      Full Step   : Preheat oven to 375 degrees f (190 degrees c) and grease a 9x13-inch

  [2] Continuation: 425 degrees f (220 degrees c) and line a large
      Full Step   : Preheat oven to 425 degrees f (220 degrees c) and line a large

  [3] Continuation: 400 degrees f.
      Full Step   : Preheat oven to 400 degrees f.
=================================================================
```

### 2. Configure Suggestion Length (`-l`)
Control the length in words of the generated continuation:
```bash
# Short 4-word continuation:
python -m app.cli autocomplete "Whisk together the" -l 4

# Longer 14-word continuation:
python -m app.cli autocomplete "Whisk together the" -l 14
```

### 3. Choose Smoothing Technique (`-s`)
Select between **Laplace**, **Lidstone**, or **Backoff** smoothing:
```bash
python -m app.cli autocomplete "Melt unsalted butter in a" -s backoff
python -m app.cli autocomplete "Melt unsalted butter in a" -s lidstone
python -m app.cli autocomplete "Melt unsalted butter in a" -s laplace
```

### 4. Interactive Chef Shell (`interactive`)
An interactive REPL session for home cooks:
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

### 5. Inspect Model Stats & Perplexity (`stats`)
```bash
python -m app.cli stats
```
**Actual Output:**
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

### 6. Add Custom Recipe Steps (`add`)
```bash
python -m app.cli add "Simmer marinara sauce gently over low heat for 30 minutes."
```

---

## 📐 Mathematical Formulation & Evaluation

### 1. Maximum-Likelihood Estimation (MLE)
For context $c = (w_{i-n+1}, \dots, w_{i-1})$:
$$P_{\text{MLE}}(w_i \mid c) = \frac{C(c, w_i)}{C(c)}$$

### 2. Laplace (Add-1) Smoothing
Ensures non-zero probabilities for all vocabulary words $w \in V$:
$$P_{\text{Laplace}}(w_i \mid c) = \frac{C(c, w_i) + 1}{C(c) + |V|}$$
*(Sum over all $w \in V$ equals $1.0$.)*

### 3. Lidstone (Add-k) Smoothing
Adds pseudo-count $k \in (0, 1]$ (default $k=0.5$):
$$P_{\text{Lidstone}}(w_i \mid c) = \frac{C(c, w_i) + k}{C(c) + k|V|}$$

### 4. Stupid Backoff
Recursively falls back to $(n-1)$-grams discounted by factor $\alpha = 0.4$:
$$S(w_i \mid w_{i-k+1}^{i-1}) = \begin{cases} \frac{C(w_{i-k+1}^i)}{C(w_{i-k+1}^{i-1})} & \text{if } C(w_{i-k+1}^i) > 0 \\ \alpha \cdot S(w_i \mid w_{i-k+2}^{i-1}) & \text{otherwise} \end{cases}$$

### 5. Quantitative Model Comparison (Perplexity)
Perplexity measures how well the probability distribution predicts held-out recipe sentences (lower is better):
$$PP(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid c_i) \right)$$

On test recipe sentences:
- **Laplace Smoothing**: $PP = 260.81$
- **Lidstone Smoothing ($k=0.5$)**: $PP = 202.70$
- **Stupid Backoff**: $PP = 13.35$

### Limits & Boundary Conditions
1. **Vocabulary Coverage**: Unseen out-of-vocabulary terms are mapped to `<unk>`.
2. **Context Horizon**: A trigram model conditions on 2 preceding tokens. While highly coherent for common phrases, long-range dependencies across multiple sentences require higher $N$ or an LLM hybrid.
3. **Corpus Domain**: The model is optimized for cooking instructions; non-culinary prompts fall back to general unigram/backoff distributions.

---

## 🧪 Testing

Run the full automated test suite with `pytest`:

```bash
pytest
```

Run with verbose test case listing:
```bash
pytest -v
```

### Test Suite Structure
- `tests/test_smoke.py`: Verifies module imports, CLI `hello` command, dataset loading, and LLM fallback handling.
- `tests/test_core.py`: 37 unit tests covering:
  - Text preprocessing (vulgar fractions, quotes, dashes).
  - Culinary tokenization (contractions, fractions, punctuation).
  - N-gram count validation and MLE probabilities.
  - Probability distribution sums ($\sum P = 1.0$) for Laplace and Lidstone.
  - Backoff smoothing hierarchy.
  - Next-word ranking and greedy/temperature sampling.
  - Continuation length and stopping criteria.
  - Model serialization and deserialization.
  - CLI argument handling and execution.

All 37 tests pass with zero failures.
