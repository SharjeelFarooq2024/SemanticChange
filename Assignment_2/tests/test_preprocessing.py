"""Tests for preprocessing and tokenization."""
from __future__ import annotations

from Assignment_2.src.preprocessing.text_preprocessor import TextPreprocessor
from Assignment_2.src.preprocessing.tokenizer import Tokenizer


# --------------------------------------------------------------------------- #
# Preprocessor
# --------------------------------------------------------------------------- #
def test_whitespace_is_collapsed_and_ends_stripped():
    pre = TextPreprocessor()
    assert pre.process("  a   b \n\t c  ") == "a b c"


def test_none_and_empty_inputs_are_safe():
    pre = TextPreprocessor()
    assert pre.process(None) == ""
    assert pre.process("") == ""


def test_control_characters_are_removed():
    pre = TextPreprocessor()
    assert pre.process("abc\x00\x1fdef") == "abc def"


def test_original_input_string_is_not_mutated():
    original = "  spaced   out  "
    TextPreprocessor().process(original)
    assert original == "  spaced   out  "


def test_lowercase_is_off_by_default():
    pre = TextPreprocessor()
    assert pre.process("Året 1883") == "Året 1883"
    assert TextPreprocessor({"lowercase": True}).process("Året 1883") == "året 1883"


def test_preprocessor_describe_reflects_config():
    pre = TextPreprocessor({"lowercase": True})
    assert pre.describe()["lowercase"] is True


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
def test_swedish_characters_are_preserved():
    tokens = Tokenizer().tokenize("fröken och förorten")
    assert tokens == ["fröken", "och", "förorten"]


def test_digits_and_years_are_preserved():
    # Temporal signal must survive tokenization.
    tokens = Tokenizer().tokenize("perioden 1880-1883 och år 2023")
    assert "1880" in tokens and "1883" in tokens and "2023" in tokens


def test_punctuation_is_a_separator():
    tokens = Tokenizer().tokenize('sade: "hej!", sedan gick han.')
    assert tokens == ["sade", "hej", "sedan", "gick", "han"]


def test_lowercase_setting_is_configurable():
    assert Tokenizer({"lowercase": True}).tokenize("Fru") == ["fru"]
    assert Tokenizer({"lowercase": False}).tokenize("Fru") == ["Fru"]


def test_empty_text_yields_no_tokens():
    assert Tokenizer().tokenize("") == []
    assert Tokenizer().tokenize(None) == []


def test_query_and_document_use_identical_tokenization():
    # Same tokenizer instance processes both sides identically.
    tok = Tokenizer({"lowercase": True})
    text = "Kvinnan i staden"
    assert tok.tokenize(text) == tok.tokenize(text.lower()) == tok.tokenize("  " + text + " ")
