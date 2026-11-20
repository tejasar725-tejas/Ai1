"""Unit tests for Recipe Step Autocompleter core algorithms and models.

Covers:
- Milestone 1: Text preprocessing, tokenization, sequence padding, and detokenization.
- Milestone 2: N-gram counting, Maximum Likelihood Estimation (MLE),
  Laplace (add-1) smoothing, Lidstone (add-k) smoothing, and Backoff smoothing.
- Milestone 3: Next-word prediction, sampling strategies (greedy vs temperature),
  recipe step continuation generation, and stopping criteria.
- Stretch Goals: Multiple suggestions, configurable length, smoothing technique comparison,
  and perplexity evaluation.
- Storage: Serialization and deserialization of model weights/counts.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

from app.core import (
    END_TOKEN,
    START_TOKEN,
    UNK_TOKEN,
    NGramModel,
    detokenize,
    pad_sequence,
    preprocess_text,
    tokenize,
)
from app.storage import (
    append_recipe_step,
    load_model_artifact,
    load_recipe_corpus,
    save_model_artifact,
)


class TestPreprocessingAndTokenization:
    """Milestone 1: Text preprocessing and tokenization tests."""

    def test_preprocess_vulgar_fractions(self):
        """Vulgar fraction unicode symbols should convert to ascii fractions."""
        raw = "Add ½ tsp of salt and ¾ cup of whole milk."
        cleaned = preprocess_text(raw, lowercase=True)
        assert "1/2" in cleaned
        assert "3/4" in cleaned
        assert "½" not in cleaned
        assert "¾" not in cleaned

    def test_preprocess_quotes_and_dashes(self):
        """Curly quotes and em/en dashes should be standardized."""
        raw = "“Sear the steak”—then remove from heat."
        cleaned = preprocess_text(raw, lowercase=True)
        assert '"sear the steak"' in cleaned
        assert "—" not in cleaned

    def test_tokenize_preserves_fractions(self):
        """Tokenize should keep fractions as unified single tokens."""
        text = "Whisk 1/2 cup flour and 3/4 tsp sugar."
        tokens = tokenize(text)
        assert "1/2" in tokens
        assert "3/4" in tokens

    def test_tokenize_preserves_contractions(self):
        """Tokenize should preserve contractions like don't or it's."""
        text = "Don't overmix the batter."
        tokens = tokenize(text)
        assert "don't" in tokens or ("don" in tokens and "n't" in tokens)
        assert "overmix" in tokens

    def test_tokenize_handles_punctuation(self):
        """Punctuation marks should separate cleanly from neighboring words."""
        text = "Heat oil in skillet. Add garlic, onions, and salt!"
        tokens = tokenize(text)
        assert "." in tokens
        assert "," in tokens
        assert "!" in tokens
        assert "skillet" in tokens
        assert "garlic" in tokens

    def test_tokenize_empty_and_whitespace(self):
        """Empty or whitespace-only inputs should return empty token lists."""
        assert tokenize("") == []
        assert tokenize("   \n\t  ") == []

    def test_detokenize_sentence_formatting(self):
        """Detokenize should format spacing and capitalization naturally."""
        tokens = ["heat", "olive", "oil", "in", "a", "skillet", "."]
        reconstructed = detokenize(tokens)
        assert reconstructed == "Heat olive oil in a skillet."
        # Verify comma spacing
        tokens_with_comma = ["add", "garlic", ",", "onions", ",", "and", "oil", "."]
        assert detokenize(tokens_with_comma) == "Add garlic, onions, and oil."

    def test_detokenize_strips_special_boundary_tokens(self):
        """Special boundary tokens <s>, </s>, <unk> should not appear in final output."""
        tokens = [START_TOKEN, START_TOKEN, "stir", "well", END_TOKEN]
        result = detokenize(tokens)
        assert START_TOKEN not in result
        assert END_TOKEN not in result
        assert result == "Stir well"

    def test_pad_sequence_orders(self):
        """Verify pad_sequence correctly prepends (n-1) start tokens and 1 end token."""
        tokens = ["bake", "the", "cake"]
        # n = 3 (trigram)
        padded_3 = pad_sequence(tokens, n=3)
        assert padded_3 == [START_TOKEN, START_TOKEN, "bake", "the", "cake", END_TOKEN]

        # n = 2 (bigram)
        padded_2 = pad_sequence(tokens, n=2)
        assert padded_2 == [START_TOKEN, "bake", "the", "cake", END_TOKEN]

        # n = 1 (unigram)
        padded_1 = pad_sequence(tokens, n=1)
        assert padded_1 == ["bake", "the", "cake", END_TOKEN]


class TestNGramCountingAndMLE:
    """Milestone 2: N-gram counting and Maximum-Likelihood Estimation tests."""

    @pytest.fixture
    def mini_model(self) -> NGramModel:
        """Create and train an n-gram model on a deterministic mini corpus."""
        corpus = [
            "heat olive oil in skillet.",
            "heat olive oil in pan.",
            "heat vegetable oil in pan.",
        ]
        model = NGramModel(n=3, unk_threshold=1)
        model.train(corpus)
        return model

    def test_ngram_counting_accuracy(self, mini_model: NGramModel):
        """Verify n-gram and context counts match expected occurrences."""
        # 'heat olive oil' appeared twice
        ctx_heat_olive = ("heat", "olive")
        assert mini_model.ngram_counts[3][ctx_heat_olive]["oil"] == 2
        assert mini_model.context_counts[3][ctx_heat_olive] == 2

        # 'heat' appears 3 times in total
        ctx_unigram = ()
        assert mini_model.ngram_counts[1][ctx_unigram]["heat"] == 3

    def test_mle_probabilities(self, mini_model: NGramModel):
        """P_MLE should match exact relative frequencies."""
        # Following ('heat', 'olive'), 'oil' occurs 2/2 times = 1.0
        prob_oil = mini_model.probability_mle("oil", ("heat", "olive"))
        assert prob_oil == 1.0

        # Following ('oil', 'in'), 'skillet' occurs 1 time, 'pan' occurs 2 times
        prob_skillet = mini_model.probability_mle("skillet", ("oil", "in"))
        prob_pan = mini_model.probability_mle("pan", ("oil", "in"))
        assert pytest.approx(prob_skillet) == 1.0 / 3.0
        assert pytest.approx(prob_pan) == 2.0 / 3.0
        assert pytest.approx(prob_skillet + prob_pan) == 1.0

    def test_mle_unseen_context_or_word(self, mini_model: NGramModel):
        """P_MLE for unseen words or contexts should be 0.0."""
        prob_unseen_word = mini_model.probability_mle("chocolate", ("heat", "olive"))
        assert prob_unseen_word == 0.0

        prob_unseen_context = mini_model.probability_mle("oil", ("frost", "cupcake"))
        assert prob_unseen_context == 0.0


class TestSmoothingTechniques:
    """Milestone 2 & Stretch Goals: Laplace, Lidstone, and Backoff smoothing."""

    @pytest.fixture
    def mini_model(self) -> NGramModel:
        corpus = [
            "bring water to a boil.",
            "bring broth to a simmer.",
        ]
        model = NGramModel(n=3, unk_threshold=1)
        model.train(corpus)
        return model

    def test_laplace_smoothing_formula(self, mini_model: NGramModel):
        """Verify Laplace formula: P = (C + 1) / (N + |V|)."""
        ctx = ("bring", "water")
        v_size = mini_model.vocab_size
        ctx_count = mini_model.context_counts[3][ctx]  # 1 ("to")
        word_count = mini_model.ngram_counts[3][ctx]["to"]  # 1

        expected_prob = (word_count + 1.0) / (ctx_count + v_size)
        actual_prob = mini_model.probability_laplace("to", ctx)
        assert pytest.approx(actual_prob) == expected_prob

    def test_laplace_unseen_has_positive_probability(self, mini_model: NGramModel):
        """Unseen word in context must have non-zero probability under Laplace smoothing."""
        ctx = ("bring", "water")
        unseen_prob = mini_model.probability_laplace("chocolate", ctx)
        assert unseen_prob > 0.0
        v_size = mini_model.vocab_size
        ctx_count = mini_model.context_counts[3][ctx]
        assert pytest.approx(unseen_prob) == 1.0 / (ctx_count + v_size)

    def test_laplace_distribution_sums_to_one(self, mini_model: NGramModel):
        """Probabilities across all vocabulary items under Laplace smoothing must sum to 1.0."""
        ctx = ("bring", "water")
        total_p = sum(mini_model.probability_laplace(w, ctx) for w in mini_model.vocabulary)
        assert pytest.approx(total_p, abs=1e-6) == 1.0

    def test_lidstone_smoothing_sums_to_one(self, mini_model: NGramModel):
        """Lidstone smoothing (Add-k) must also sum to 1.0 across vocabulary."""
        ctx = ("to", "a")
        for k in (0.1, 0.5, 0.75):
            total_p = sum(mini_model.probability_lidstone(w, ctx, k_param=k) for w in mini_model.vocabulary)
            assert pytest.approx(total_p, abs=1e-6) == 1.0

    def test_lidstone_equals_laplace_when_k_is_one(self, mini_model: NGramModel):
        """When k=1.0, Lidstone smoothing must exactly equal Laplace smoothing."""
        ctx = ("to", "a")
        for w in mini_model.vocabulary:
            p_lap = mini_model.probability_laplace(w, ctx)
            p_lid = mini_model.probability_lidstone(w, ctx, k_param=1.0)
            assert pytest.approx(p_lap) == p_lid

    def test_backoff_smoothing_hierarchy(self, mini_model: NGramModel):
        """Backoff smoothing should leverage higher-order then lower-order models."""
        # Seen in trigram: ('to', 'a') -> 'boil' (count 1)
        prob_seen = mini_model.probability_backoff("boil", ("to", "a"), alpha=0.4)
        assert prob_seen > 0.0

        # Unseen in trigram but seen in bigram/unigram:
        prob_backed_off = mini_model.probability_backoff("broth", ("to", "a"), alpha=0.4)
        assert prob_backed_off > 0.0
        # Seen trigram should have higher score than backed-off token
        assert prob_seen > prob_backed_off


class TestPredictionAndGeneration:
    """Milestone 3: Next-word prediction and recipe continuation generation."""

    @pytest.fixture
    def trained_recipe_model(self) -> NGramModel:
        corpus = [
            "Preheat oven to 375 degrees F and grease baking dish.",
            "Preheat oven to 400 degrees F and line baking sheet.",
            "Heat olive oil in a skillet over medium heat.",
            "Add minced garlic and onions, cooking until soft.",
        ]
        model = NGramModel(n=3)
        model.train(corpus)
        return model

    def test_predict_next_words_ranking(self, trained_recipe_model: NGramModel):
        """predict_next_words should return top words sorted descending by probability."""
        preds = trained_recipe_model.predict_next_words("preheat oven to", top_k=5)
        assert len(preds) <= 5
        # Probabilities must be sorted descending
        probs = [p[1] for p in preds]
        assert probs == sorted(probs, reverse=True)
        # 375 and 400 should be the top candidates
        top_words = [p[0] for p in preds[:2]]
        assert "375" in top_words
        assert "400" in top_words

    def test_sample_next_word_greedy(self, trained_recipe_model: NGramModel):
        """With temperature=0.0 (greedy), sampling should be deterministic."""
        word1 = trained_recipe_model.sample_next_word("heat olive", temperature=0.0)
        word2 = trained_recipe_model.sample_next_word("heat olive", temperature=0.0)
        assert word1 == word2
        assert word1 == "oil"

    def test_generate_continuation_structure(self, trained_recipe_model: NGramModel):
        """generate_continuation should return dictionary with expected keys."""
        res = trained_recipe_model.generate_continuation("Heat olive oil", max_words=6)
        assert "prompt" in res
        assert "continuation" in res
        assert "full_step" in res
        assert "tokens" in res
        assert isinstance(res["continuation"], str)
        assert "in a skillet" in res["continuation"] or "skillet" in res["continuation"]

    def test_generate_continuation_respects_max_words(self, trained_recipe_model: NGramModel):
        """Generated continuation tokens should not exceed max_words."""
        for limit in (2, 4, 8):
            res = trained_recipe_model.generate_continuation("Preheat oven to", max_words=limit)
            assert len(res["tokens"]) <= limit


class TestStretchGoals:
    """Stretch Ideas: Multiple suggestions, length specification, perplexity."""

    @pytest.fixture
    def rich_model(self) -> NGramModel:
        corpus = load_recipe_corpus()
        model = NGramModel(n=3)
        model.train(corpus)
        return model

    def test_multiple_distinct_suggestions(self, rich_model: NGramModel):
        """generate_suggestions should provide multiple unique continuations."""
        prompt = "Heat olive oil in a"
        suggestions = rich_model.generate_suggestions(prompt, num_suggestions=3, length=8)
        assert len(suggestions) >= 2
        # Continuations should be distinct
        cont_texts = [s["continuation"] for s in suggestions]
        assert len(set(cont_texts)) == len(cont_texts)

    def test_user_specified_suggestion_length(self, rich_model: NGramModel):
        """User can specify shorter or longer autocomplete lengths."""
        short_res = rich_model.generate_continuation("Melt unsalted butter", max_words=3)
        long_res = rich_model.generate_continuation("Melt unsalted butter", max_words=12)
        assert len(short_res["tokens"]) <= 3
        assert len(long_res["tokens"]) <= 12

    def test_perplexity_comparison(self, rich_model: NGramModel):
        """Model perplexity can be computed across different smoothing techniques."""
        test_sentences = [
            "Heat olive oil in a large skillet.",
            "Bring a large pot of water to a boil.",
        ]
        pp_laplace = rich_model.calculate_perplexity(test_sentences, method="laplace")
        pp_lidstone = rich_model.calculate_perplexity(test_sentences, method="lidstone")
        pp_backoff = rich_model.calculate_perplexity(test_sentences, method="backoff")

        assert pp_laplace > 0.0 and not math.isinf(pp_laplace)
        assert pp_lidstone > 0.0 and not math.isinf(pp_lidstone)
        assert pp_backoff > 0.0 and not math.isinf(pp_backoff)
        # Backoff typically yields lower perplexity on seen language structures
        assert pp_backoff < pp_laplace


class TestStorageAndPersistence:
    """Storage, serialization, and corpus extension tests."""

    def test_model_serialization_roundtrip(self):
        """Model should serialize to dict/JSON and reload with identical predictions."""
        corpus = ["Heat olive oil in skillet.", "Bake cake in oven."]
        m1 = NGramModel(n=2).train(corpus)
        data = m1.to_dict()

        m2 = NGramModel.from_dict(data)
        assert m2.n == m1.n
        assert m2.vocab_size == m1.vocab_size
        assert m2.total_tokens == m1.total_tokens

        pred1 = m1.predict_next_words("heat", top_k=3)
        pred2 = m2.predict_next_words("heat", top_k=3)
        assert pred1 == pred2

    def test_save_and_load_model_artifact(self, tmp_path: Path):
        """Model artifact file save and load."""
        model = NGramModel(n=2).train(["Cook pasta until al dente."])
        artifact_path = tmp_path / "model_artifact.json"

        save_model_artifact(model.to_dict(), artifact_path)
        assert artifact_path.exists()

        loaded_data = load_model_artifact(artifact_path)
        restored_model = NGramModel.from_dict(loaded_data)
        assert restored_model.n == 2

    def test_append_recipe_step_and_load(self, tmp_path: Path):
        """Appending recipe instructions to file should be reflected upon reload."""
        test_file = tmp_path / "custom_recipes.txt"
        append_recipe_step("Chop fresh cilantro finely.", target_file=test_file)
        append_recipe_step("Squeeze fresh lime juice over tacos.", target_file=test_file)

        loaded = load_recipe_corpus(test_file)
        assert len(loaded) == 2
        assert "Chop fresh cilantro finely." in loaded
        assert "Squeeze fresh lime juice over tacos." in loaded


class TestCLIOperations:
    """End-to-end tests for CLI commands and argument configurations."""

    def test_cli_autocomplete_basic(self, capsys):
        from unittest.mock import patch
        import app.cli

        with patch.object(sys, "argv", ["app.cli", "autocomplete", "Heat olive oil in a"]):
            code = app.cli.main()
            assert code == 0

        out = capsys.readouterr().out
        assert "Input Partial Step" in out
        assert "Generated Continuation" in out

    def test_cli_autocomplete_options(self, capsys):
        from unittest.mock import patch
        import app.cli

        with patch.object(
            sys,
            "argv",
            [
                "app.cli",
                "autocomplete",
                "Preheat oven to",
                "-k",
                "3",
                "-l",
                "8",
                "-s",
                "lidstone",
            ],
        ):
            code = app.cli.main()
            assert code == 0

        out = capsys.readouterr().out
        assert "Top 3 Suggestions" in out or "Suggestions:" in out

    def test_cli_stats_command(self, capsys):
        from unittest.mock import patch
        import app.cli

        with patch.object(sys, "argv", ["app.cli", "stats"]):
            code = app.cli.main()
            assert code == 0

        out = capsys.readouterr().out
        assert "Recipe Corpus & N-Gram Model Statistics" in out
        assert "Perplexity Evaluation" in out

    def test_cli_add_command(self, tmp_path: Path, capsys):
        from unittest.mock import patch
        import app.cli

        target = tmp_path / "custom.txt"
        with patch.object(
            sys,
            "argv",
            [
                "app.cli",
                "add",
                "Simmer marinara sauce for 30 minutes.",
                "--corpus",
                str(target),
            ],
        ):
            code = app.cli.main()
            assert code == 0

        assert target.exists()
        assert "Simmer marinara sauce for 30 minutes." in target.read_text(encoding="utf-8")

