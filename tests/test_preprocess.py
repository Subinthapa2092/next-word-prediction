import numpy as np
import pytest

import config
from src.dataset import make_windows, prepare_data
from src.preprocess import Vocabulary, clean_text, split_sentences, strip_gutenberg, strip_social_noise, tokenize


def test_clean_text_lowercases_and_normalises_quotes():
    assert clean_text("  Don\u2019t  PANIC \n now ") == "don't panic now"


def test_strip_social_noise_removes_urls_mentions_and_retweet_prefix():
    tweet = "RT @someuser: can't wait for the weekend!! #excited http://t.co/abc123"
    assert strip_social_noise(tweet).strip() == "can't wait for the weekend!! excited"


def test_strip_social_noise_unwraps_hashtag_keeping_the_word():
    assert strip_social_noise("so ready for #monday already") == "so ready for monday already"


def test_strip_social_noise_is_a_no_op_on_plain_prose():
    prose = "The city council announced a new budget plan."
    assert strip_social_noise(prose) == prose


def test_clean_text_applies_social_noise_stripping():
    assert clean_text("check this out www.example.com/page @friend") == "check this out"


def test_tokenize_keeps_apostrophes_and_drops_punctuation_and_digits():
    assert tokenize("It's 221B Baker Street, isn't it?") == ["it's", "b", "baker", "street", "isn't", "it"]


def test_split_sentences_drops_tiny_fragments():
    sents = split_sentences("Hello there my friend. Ok. How are you today?")
    assert sents == [["hello", "there", "my", "friend"], ["how", "are", "you", "today"]]


def test_strip_gutenberg_removes_header_and_footer():
    raw = "junk\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nreal text\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense"
    assert strip_gutenberg(raw).strip() == "real text"


def test_vocabulary_reserves_special_ids_and_maps_unknowns():
    vocab = Vocabulary.build([["a", "a", "b"], ["a", "b", "c"]], min_freq=2)
    assert vocab.itos[config.PAD_ID] == config.PAD_TOKEN
    assert vocab.itos[config.UNK_ID] == config.UNK_TOKEN
    assert vocab.encode(["a", "zzz"]) == [vocab.stoi["a"], config.UNK_ID]
    assert "c" not in vocab.stoi  # below min_freq


def test_vocabulary_roundtrip_json(tmp_path):
    vocab = Vocabulary.build([["x", "y", "x", "y"]], min_freq=1)
    vocab.save(tmp_path / "v.json")
    loaded = Vocabulary.load(tmp_path / "v.json")
    assert loaded.itos == vocab.itos and loaded.stoi == vocab.stoi


def test_make_windows_left_pads_and_targets_next_word():
    x, y = make_windows([[5, 6, 7, 8]], seq_len=3)
    assert x.tolist() == [[0, 0, 5], [0, 5, 6], [5, 6, 7]]
    assert y.tolist() == [6, 7, 8]


def test_make_windows_uses_only_last_seq_len_words():
    x, y = make_windows([[1, 2, 3, 4, 5, 6]], seq_len=2)
    assert x[-1].tolist() == [4, 5] and y[-1] == 6


def test_make_windows_handles_empty_input():
    x, y = make_windows([[3]], seq_len=4)
    assert x.shape == (0, 4) and y.shape == (0,)


def test_prepare_data_shapes_and_no_vocab_leakage(corpus_text):
    s = prepare_data(corpus_text, seq_len=5)
    assert s.x_train.shape[1] == 5 and len(s.x_train) == len(s.y_train)
    assert len(s.x_val) > 0 and len(s.x_test) > 0
    assert s.x_train.max() < len(s.vocab) and s.y_train.max() < len(s.vocab)


def test_prepare_data_is_deterministic(corpus_text):
    a, b = prepare_data(corpus_text), prepare_data(corpus_text)
    assert np.array_equal(a.x_train, b.x_train) and a.vocab.itos == b.vocab.itos


def test_prepare_data_rejects_tiny_corpus():
    with pytest.raises(ValueError):
        prepare_data("Just one sentence here.")
