"""Core NLP and N-Gram Language Model module for the Recipe Step Autocompleter.

Implements:
- Milestone 1: Recipe text preprocessing, cleaning, and tokenization.
- Milestone 2: N-gram counting, Maximum-Likelihood Estimation (MLE),
  Laplace (add-1) smoothing, Lidstone (add-k) smoothing, and Backoff smoothing.
- Milestone 3: Next-word prediction, sampling strategies (greedy, temperature, top-k),
  recipe step continuation generation, and multi-candidate suggestions.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


# Special boundary and out-of-vocabulary tokens
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# Punctuation regex for tokenization
PUNCTUATION_REGEX = re.compile(r"([.,!?;:()\[\]\"'\-/])")
# Regex for matching fraction numbers like 1/2, 3/4
FRACTION_MAP = {
    "½": "1/2",
    "⅓": "1/3",
    "⅔": "2/3",
    "¼": "1/4",
    "¾": "3/4",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}


def preprocess_text(text: str, lowercase: bool = True) -> str:
    """Clean and normalize raw recipe instruction text.

    - Replaces unicode vulgar fractions with ascii equivalents (e.g., ½ -> 1/2)
    - Normalizes curly quotes and special dashes
    - Strips leading/trailing whitespace
    - Lowercases if requested
    """
    if not text:
        return ""

    cleaned = text
    # Replace fractions
    for frac_char, ascii_frac in FRACTION_MAP.items():
        cleaned = cleaned.replace(frac_char, ascii_frac)

    # Replace curly quotes and apostrophes
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    cleaned = cleaned.replace("‘", "'").replace("’", "'")
    cleaned = cleaned.replace("—", " - ").replace("–", " - ")

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if lowercase:
        cleaned = cleaned.lower()

    return cleaned


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Tokenize recipe text into clean word and punctuation tokens.

    Preserves:
    - Fractions like 1/2, 3/4
    - Numbers with units or ranges like 9x13-inch, 15-20, 3-4
    - Contractions like don't, it's
    """
    normalized = preprocess_text(text, lowercase=lowercase)
    if not normalized:
        return []

    # Preserve fractions (e.g., 1/2, 3/4) by protecting them temporarily
    protected_tokens: List[str] = []

    def _preserve_token(match: re.Match) -> str:
        idx = len(protected_tokens)
        protected_tokens.append(match.group(0))
        return f"__PROT_{idx}__"

    # Protect fractions like 1/2, 3/4
    subbed = re.sub(r"\b\d+/\d+\b", _preserve_token, normalized)
    # Protect contractions like don't, it's, chef's
    subbed = re.sub(r"\b[a-zA-Z]+'[a-zA-Z]+\b", _preserve_token, subbed)

    # Separate punctuation with spaces (excluding apostrophe inside protected tokens)
    subbed = re.sub(r"([.,!?;:()\"'])", r" \1 ", subbed)

    raw_tokens = subbed.split()
    tokens: List[str] = []

    for t in raw_tokens:
        t = t.strip()
        if not t:
            continue
        # Restore protected tokens
        if t.startswith("__PROT_") and t.endswith("__"):
            try:
                prot_idx = int(t[7:-2])
                tokens.append(protected_tokens[prot_idx])
                continue
            except (ValueError, IndexError):
                pass
        tokens.append(t)

    return tokens


def detokenize(tokens: Sequence[str]) -> str:
    """Convert tokens back into natural, readable recipe text.

    Removes special boundary tags (`<s>`, `</s>`, `<unk>`),
    handles punctuation spacing properly, and capitalizes the first character.
    """
    if not tokens:
        return ""

    filtered = [t for t in tokens if t not in (START_TOKEN, END_TOKEN, UNK_TOKEN)]
    if not filtered:
        return ""

    text = ""
    no_space_before = {".", ",", "!", "?", ":", ";", ")", "]", "'"}
    no_space_after = {"(", "["}

    prev_token = ""
    for token in filtered:
        if not text:
            text = token
        elif token in no_space_before or prev_token in no_space_after:
            text += token
        elif token == "'" or (prev_token and prev_token.endswith("'")):
            text += token
        elif prev_token == "n't" or token == "n't":
            text += token
        else:
            text += " " + token
        prev_token = token

    # Capitalize the first letter if it's alphanumeric
    if text:
        text = text[0].upper() + text[1:]

    return text


def pad_sequence(tokens: Sequence[str], n: int) -> List[str]:
    """Pad token sequence with (n-1) start tokens and 1 end token.

    For n=3:
    ['heat', 'olive', 'oil'] -> ['<s>', '<s>', 'heat', 'olive', 'oil', '</s>']
    """
    if n <= 1:
        return list(tokens) + [END_TOKEN]
    start_padding = [START_TOKEN] * (n - 1)
    return start_padding + list(tokens) + [END_TOKEN]


class NGramModel:
    """N-gram statistical language model with MLE and smoothing techniques.

    Supports:
    - Configurable order n (e.g. n=1, 2, 3, 4)
    - Maximum-Likelihood Estimation (MLE)
    - Laplace (Add-1) smoothing
    - Lidstone (Add-k) smoothing
    - Backoff (Stupid Backoff / Lower-order fallbacks)
    - Next-word distribution prediction and sampling
    - Step continuation generation and multi-phrase suggestions
    - Perplexity evaluation
    """

    def __init__(self, n: int = 3, unk_threshold: int = 1):
        """Initialize the N-gram language model.

        Args:
            n: Order of the n-gram model (default 3 for trigram).
            unk_threshold: Minimum word count threshold below which words are treated as UNK.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        self.n = n
        self.unk_threshold = unk_threshold

        # N-gram count tables:
        # ngram_counts[order][context_tuple][word] = count
        # For order=1: context is ()
        self.ngram_counts: Dict[int, Dict[Tuple[str, ...], Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        self.context_counts: Dict[int, Counter[Tuple[str, ...]]] = defaultdict(Counter)

        # Total number of tokens trained on
        self.total_tokens: int = 0
        # Vocabulary set (includes special tokens)
        self.vocabulary: Set[str] = set()
        # Raw word frequencies for unk filtering
        self.raw_word_counts: Counter[str] = Counter()
        self._is_trained: bool = False

    @property
    def vocab_size(self) -> int:
        """Size of the model vocabulary |V|."""
        return len(self.vocabulary)

    def train(self, corpus: Iterable[str]) -> "NGramModel":
        """Train the model on a corpus of recipe instruction steps.

        Args:
            corpus: Iterable of recipe instruction strings.

        Returns:
            self for method chaining.
        """
        tokenized_sentences: List[List[str]] = []
        self.raw_word_counts.clear()

        # Step 1: Tokenize all sentences and count vocabulary
        for sentence in corpus:
            tokens = tokenize(sentence)
            if tokens:
                tokenized_sentences.append(tokens)
                for t in tokens:
                    self.raw_word_counts[t] += 1

        # Step 2: Establish vocabulary with UNK threshold
        self.vocabulary = {
            word for word, count in self.raw_word_counts.items() if count >= self.unk_threshold
        }
        self.vocabulary.add(START_TOKEN)
        self.vocabulary.add(END_TOKEN)
        self.vocabulary.add(UNK_TOKEN)

        # Step 3: Count n-grams of all orders from 1 up to self.n
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.total_tokens = 0

        for tokens in tokenized_sentences:
            # Replace rare words with UNK
            clean_tokens = [t if t in self.vocabulary else UNK_TOKEN for t in tokens]
            self.total_tokens += len(clean_tokens)

            # Pad sequence according to max order self.n
            padded = pad_sequence(clean_tokens, self.n)

            # Collect counts for all orders k from 1 to self.n
            for k in range(1, self.n + 1):
                ctx_len = k - 1
                for i in range(ctx_len, len(padded)):
                    if ctx_len == 0:
                        context: Tuple[str, ...] = ()
                    else:
                        context = tuple(padded[i - ctx_len : i])
                    word = padded[i]
                    self.ngram_counts[k][context][word] += 1
                    self.context_counts[k][context] += 1

        self._is_trained = True
        return self

    def _normalize_context(self, context: Sequence[str], order: int) -> Tuple[str, ...]:
        """Convert arbitrary sequence to appropriate context tuple of length (order - 1)."""
        ctx_len = order - 1
        if ctx_len == 0:
            return ()

        # Map words to UNK if not in vocabulary
        clean_ctx = [w if w in self.vocabulary else UNK_TOKEN for w in context]

        if len(clean_ctx) < ctx_len:
            # Left-pad with <s>
            clean_ctx = [START_TOKEN] * (ctx_len - len(clean_ctx)) + clean_ctx
        elif len(clean_ctx) > ctx_len:
            # Take last ctx_len tokens
            clean_ctx = clean_ctx[-ctx_len:]

        return tuple(clean_ctx)

    def probability_mle(self, word: str, context: Sequence[str], order: Optional[int] = None) -> float:
        """Calculate Maximum-Likelihood Probability P_MLE(word | context).

        Formula: P(w | c) = C(c, w) / C(c)
        """
        k = order or self.n
        ctx = self._normalize_context(context, k)
        target = word if word in self.vocabulary else UNK_TOKEN

        ctx_count = self.context_counts[k][ctx]
        if ctx_count == 0:
            return 0.0

        word_count = self.ngram_counts[k][ctx][target]
        return word_count / ctx_count

    def probability_laplace(self, word: str, context: Sequence[str], order: Optional[int] = None) -> float:
        """Calculate Laplace (Add-1) smoothed probability.

        Formula: P_Laplace(w | c) = (C(c, w) + 1) / (C(c) + |V|)
        """
        k = order or self.n
        ctx = self._normalize_context(context, k)
        target = word if word in self.vocabulary else UNK_TOKEN

        ctx_count = self.context_counts[k][ctx]
        word_count = self.ngram_counts[k][ctx][target]
        v_size = max(self.vocab_size, 1)

        return (word_count + 1.0) / (ctx_count + v_size)

    def probability_lidstone(
        self, word: str, context: Sequence[str], k_param: float = 0.5, order: Optional[int] = None
    ) -> float:
        """Calculate Lidstone (Add-k) smoothed probability.

        Formula: P_Lidstone(w | c) = (C(c, w) + k) / (C(c) + k * |V|)
        """
        if k_param <= 0:
            return self.probability_mle(word, context, order)

        k = order or self.n
        ctx = self._normalize_context(context, k)
        target = word if word in self.vocabulary else UNK_TOKEN

        ctx_count = self.context_counts[k][ctx]
        word_count = self.ngram_counts[k][ctx][target]
        v_size = max(self.vocab_size, 1)

        return (word_count + k_param) / (ctx_count + k_param * v_size)

    def probability_backoff(
        self, word: str, context: Sequence[str], alpha: float = 0.4, order: Optional[int] = None
    ) -> float:
        """Calculate probability using Stupid Backoff smoothing.

        Recursively checks higher-order n-grams; if zero count, backs off
        to (k-1)-gram multiplied by discount factor alpha.
        """
        max_order = order or self.n
        target = word if word in self.vocabulary else UNK_TOKEN

        factor = 1.0
        for k in range(max_order, 0, -1):
            ctx = self._normalize_context(context, k)
            ctx_count = self.context_counts[k][ctx]
            word_count = self.ngram_counts[k][ctx][target]

            if word_count > 0 and ctx_count > 0:
                return factor * (word_count / ctx_count)
            factor *= alpha

        # Fallback to uniform / minimum non-zero
        v_size = max(self.vocab_size, 1)
        return factor * (1.0 / v_size)

    def get_word_probability(
        self,
        word: str,
        context: Sequence[str],
        method: str = "laplace",
        order: Optional[int] = None,
        lidstone_k: float = 0.5,
        backoff_alpha: float = 0.4,
    ) -> float:
        """Get word probability under the specified smoothing method."""
        method_lower = method.lower()
        if method_lower == "laplace":
            return self.probability_laplace(word, context, order)
        elif method_lower == "lidstone":
            return self.probability_lidstone(word, context, k_param=lidstone_k, order=order)
        elif method_lower in ("backoff", "stupid_backoff"):
            return self.probability_backoff(word, context, alpha=backoff_alpha, order=order)
        elif method_lower == "mle":
            return self.probability_mle(word, context, order)
        else:
            raise ValueError(f"Unknown smoothing method: {method}. Choose from laplace, lidstone, backoff, mle.")

    def predict_next_words(
        self,
        context: Sequence[str] | str,
        top_k: int = 5,
        method: str = "laplace",
        exclude_special: bool = False,
        order: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Return the top_k most probable next words given context.

        Args:
            context: Text prompt string or list of context tokens.
            top_k: Number of candidate predictions to return.
            method: Smoothing method (laplace, lidstone, backoff, mle).
            exclude_special: If True, filters out <s> and <unk> from suggestions.
            order: N-gram order override.

        Returns:
            Sorted list of (word, probability) tuples descending by probability.
        """
        if isinstance(context, str):
            ctx_tokens = tokenize(context)
        else:
            ctx_tokens = list(context)

        # Candidates to score: all words in vocabulary
        scores: List[Tuple[str, float]] = []
        for word in self.vocabulary:
            if exclude_special and word in (START_TOKEN, UNK_TOKEN):
                continue
            prob = self.get_word_probability(word, ctx_tokens, method=method, order=order)
            scores.append((word, prob))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]

    def sample_next_word(
        self,
        context: Sequence[str] | str,
        method: str = "laplace",
        temperature: float = 0.0,
        top_k: int = 10,
        exclude_special: bool = True,
        order: Optional[int] = None,
    ) -> str:
        """Sample the next word from the distribution.

        Args:
            context: Text prompt string or list of tokens.
            method: Smoothing technique.
            temperature: Sampling temperature.
                         0.0 = greedy (argmax),
                         > 0.0 = stochastic temperature sampling.
            top_k: Restrict sampling to top_k candidates for quality.
            exclude_special: Exclude <s> and <unk> from candidate generation.
            order: Order override.

        Returns:
            The chosen next word token.
        """
        k = order or self.n
        if isinstance(context, str):
            ctx_tokens = tokenize(context)
        else:
            ctx_tokens = list(context)

        # For generation, find candidate words observed in context (with backoff if needed)
        seen_words: List[str] = []
        for check_order in range(k, 0, -1):
            ctx = self._normalize_context(ctx_tokens, check_order)
            counts_in_ctx = self.ngram_counts[check_order].get(ctx, {})
            observed = [
                w for w, count in counts_in_ctx.items()
                if count > 0 and (not exclude_special or w not in (START_TOKEN, UNK_TOKEN))
            ]
            if observed:
                seen_words = observed
                break

        # Calculate probabilities for candidate words
        candidates: List[Tuple[str, float]] = []
        if seen_words:
            for w in seen_words:
                prob = self.get_word_probability(w, ctx_tokens, method=method, order=k)
                candidates.append((w, prob))
            candidates.sort(key=lambda item: (-item[1], item[0]))
            candidates = candidates[:top_k]
        else:
            candidates = self.predict_next_words(
                ctx_tokens, top_k=max(top_k, 1), method=method, exclude_special=exclude_special, order=k
            )

        if not candidates:
            return END_TOKEN

        # Greedy sampling
        if temperature <= 0.0:
            return candidates[0][0]

        # Temperature-scaled sampling
        words = [c[0] for c in candidates]
        probs = [c[1] for c in candidates]

        # Apply temperature scaling: p_i^(1/T)
        scaled_logits = []
        for p in probs:
            safe_p = max(p, 1e-12)
            scaled_logits.append(math.log(safe_p) / temperature)

        max_logit = max(scaled_logits)
        exp_logits = [math.exp(lg - max_logit) for lg in scaled_logits]
        sum_exp = sum(exp_logits)
        if sum_exp <= 0:
            return words[0]

        norm_probs = [el / sum_exp for el in exp_logits]
        chosen = random.choices(words, weights=norm_probs, k=1)[0]
        return chosen

    def generate_continuation(
        self,
        prompt: str,
        max_words: int = 12,
        method: str = "laplace",
        temperature: float = 0.0,
        top_k: int = 10,
        stop_at_sentence_end: bool = True,
        order: Optional[int] = None,
    ) -> Dict[str, Union[str, List[str], float]]:
        """Generate a recipe step continuation given a partial recipe step prompt.

        Args:
            prompt: Partial recipe step text (e.g. "Heat olive oil in a").
            max_words: Maximum number of words to generate.
            method: Smoothing method ('laplace', 'lidstone', 'backoff', 'mle').
            temperature: 0.0 for deterministic greedy continuation, >0 for creative sampling.
            top_k: Top-k candidate truncation during sampling.
            stop_at_sentence_end: Stop when </s> or sentence terminator ('.') is reached.
            order: N-gram order override.

        Returns:
            Dictionary containing:
            - 'prompt': Original input prompt
            - 'continuation': Generated continuation string
            - 'full_step': Completed recipe instruction
            - 'tokens': List of generated tokens
        """
        prompt_tokens = tokenize(prompt)
        current_tokens = list(prompt_tokens)
        generated_tokens: List[str] = []

        for _ in range(max_words):
            # Sample next word with repetition penalty for recent cycles
            next_word = self.sample_next_word(
                current_tokens,
                method=method,
                temperature=temperature,
                top_k=top_k,
                exclude_special=True,
                order=order,
            )

            # Prevent repetitive cycle loops (e.g. repeated bigrams or trigrams in recent window)
            recent_tokens = current_tokens[-8:]
            if len(recent_tokens) >= 2:
                pair = (recent_tokens[-1], next_word)
                pairs_in_window = [
                    (recent_tokens[i], recent_tokens[i + 1])
                    for i in range(len(recent_tokens) - 2)
                ]
                if pair in pairs_in_window:
                    candidates = self.predict_next_words(current_tokens, top_k=8, method=method, exclude_special=True)
                    for alt_w, _ in candidates:
                        alt_pair = (recent_tokens[-1], alt_w)
                        if alt_pair not in pairs_in_window:
                            next_word = alt_w
                            break

            if next_word == END_TOKEN:
                break

            generated_tokens.append(next_word)
            current_tokens.append(next_word)

            if stop_at_sentence_end and next_word in (".", "!", "?") and len(generated_tokens) >= 3:
                break

        continuation_text = detokenize(generated_tokens)
        full_tokens = prompt_tokens + generated_tokens
        full_step = detokenize(full_tokens)

        return {
            "prompt": prompt,
            "continuation": continuation_text,
            "full_step": full_step,
            "tokens": generated_tokens,
        }

    def generate_suggestions(
        self,
        prompt: str,
        num_suggestions: int = 3,
        length: int = 10,
        method: str = "laplace",
        temperature: float = 0.7,
        order: Optional[int] = None,
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """Generate multiple distinct recipe step continuation suggestions (Stretch Goal).

        Args:
            prompt: Partial recipe instruction.
            num_suggestions: Number of distinct completions to provide.
            length: Length in words of the suggestions.
            method: Smoothing technique.
            temperature: Sampling temperature for variation.
            order: N-gram order.

        Returns:
            List of suggestion dictionaries with unique continuations.
        """
        suggestions: List[Dict[str, Union[str, List[str]]]] = []
        seen_continuations: Set[str] = set()

        # Always include the greedy (most likely) continuation first
        greedy_res = self.generate_continuation(
            prompt, max_words=length, method=method, temperature=0.0, order=order
        )
        if greedy_res["continuation"]:
            suggestions.append(greedy_res)
            seen_continuations.add(str(greedy_res["continuation"]).lower())

        # Sample additional distinct continuations
        attempts = 0
        max_attempts = num_suggestions * 8

        while len(suggestions) < num_suggestions and attempts < max_attempts:
            attempts += 1
            # Slight temperature variation for diversity
            effective_temp = max(temperature, 0.4) + (attempts * 0.05)
            cand = self.generate_continuation(
                prompt,
                max_words=length,
                method=method,
                temperature=effective_temp,
                top_k=8,
                order=order,
            )
            cont_str = str(cand["continuation"]).strip().lower()
            if cont_str and cont_str not in seen_continuations:
                suggestions.append(cand)
                seen_continuations.add(cont_str)

        return suggestions

    def calculate_perplexity(
        self, test_sentences: Sequence[str], method: str = "laplace", order: Optional[int] = None
    ) -> float:
        """Calculate model perplexity on a test set of recipe sentences.

        Lower perplexity indicates better predictive performance.
        Formula: PP = exp(- (1/N) * sum(log P(w_i | context)))
        """
        total_log_prob = 0.0
        n_tokens = 0
        k = order or self.n

        for sentence in test_sentences:
            tokens = tokenize(sentence)
            if not tokens:
                continue
            padded = pad_sequence(tokens, k)
            ctx_len = k - 1
            for i in range(ctx_len, len(padded)):
                ctx = padded[i - ctx_len : i] if ctx_len > 0 else ()
                word = padded[i]
                prob = self.get_word_probability(word, ctx, method=method, order=k)
                total_log_prob += math.log(max(prob, 1e-15))
                n_tokens += 1

        if n_tokens == 0:
            return float("inf")

        cross_entropy = -total_log_prob / n_tokens
        return math.exp(cross_entropy)

    def to_dict(self) -> dict:
        """Serialize model parameters and counts for saving."""
        serialized_counts = {}
        for order, contexts in self.ngram_counts.items():
            serialized_counts[str(order)] = {
                "__SEP__".join(ctx): dict(counter) for ctx, counter in contexts.items()
            }

        serialized_ctx_counts = {}
        for order, contexts in self.context_counts.items():
            serialized_ctx_counts[str(order)] = {
                "__SEP__".join(ctx): count for ctx, count in contexts.items()
            }

        return {
            "n": self.n,
            "unk_threshold": self.unk_threshold,
            "vocab": list(self.vocabulary),
            "ngram_counts": serialized_counts,
            "context_counts": serialized_ctx_counts,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        """Deserialize model from a dictionary."""
        model = cls(n=data.get("n", 3), unk_threshold=data.get("unk_threshold", 1))
        model.vocabulary = set(data.get("vocab", []))
        model.total_tokens = data.get("total_tokens", 0)

        for order_str, contexts in data.get("ngram_counts", {}).items():
            order = int(order_str)
            for ctx_key, word_counts in contexts.items():
                ctx = tuple(ctx_key.split("__SEP__")) if ctx_key else ()
                model.ngram_counts[order][ctx] = Counter(word_counts)

        for order_str, contexts in data.get("context_counts", {}).items():
            order = int(order_str)
            for ctx_key, count in contexts.items():
                ctx = tuple(ctx_key.split("__SEP__")) if ctx_key else ()
                model.context_counts[order][ctx] = count

        model._is_trained = True
        return model
