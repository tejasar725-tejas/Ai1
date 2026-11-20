"""Smoke tests for the Recipe Step Autocompleter.

These verify that all packages, modules, default corpora,
and CLI entry points are healthy and operational.
"""

import sys
from unittest.mock import patch

import pytest

import app
import app.cli
import app.core
import app.llm
import app.storage


def test_package_metadata():
    """Verify package version and expected exports."""
    assert hasattr(app, "__version__")
    assert app.__version__ == "1.0.0"
    assert hasattr(app, "NGramModel")
    assert hasattr(app, "preprocess_text")
    assert hasattr(app, "tokenize")


def test_cli_hello(capsys):
    """Verify that 'python -m app.cli hello' executes with exit code 0 and friendly output."""
    with patch.object(sys, "argv", ["app.cli", "hello"]):
        exit_code = app.cli.main()
        assert exit_code == 0

    captured = capsys.readouterr()
    assert "Recipe Step Autocompleter CLI" in captured.out
    assert "Corpus Steps" in captured.out
    assert "Smoothing" in captured.out


def test_load_builtin_corpus():
    """Verify that built-in recipe instructions load cleanly."""
    corpus = app.storage.load_recipe_corpus()
    assert isinstance(corpus, list)
    assert len(corpus) >= 50
    # Every recipe step should be a non-empty string
    assert all(isinstance(step, str) and len(step.strip()) > 0 for step in corpus)


def test_basic_ngram_pipeline():
    """Verify end-to-end flow: train on tiny corpus and generate completion."""
    tiny_corpus = [
        "Heat olive oil in a skillet over medium heat.",
        "Add garlic and onions to the skillet.",
    ]
    model = app.core.NGramModel(n=2).train(tiny_corpus)
    assert model.vocab_size > 5

    res = model.generate_continuation("Heat olive", max_words=5)
    assert "continuation" in res
    assert "full_step" in res
    assert len(res["continuation"]) > 0


def test_llm_module_fallback():
    """Verify that llm module gracefully handles calls without external services."""
    assert isinstance(app.llm.is_llm_available(), bool)
    completion = app.llm.generate_recipe_step_llm("Heat olive oil in a")
    assert isinstance(completion, str)
    assert len(completion) > 0

    hello_info = app.llm.llm_hello()
    assert "greeting" in hello_info
    assert "provider" in hello_info
