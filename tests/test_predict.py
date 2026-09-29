import pytest

from src.model import build_model


@pytest.mark.parametrize("kind", ["rnn", "lstm", "gru"])
def test_build_model_output_is_a_distribution_over_vocab(kind):
    import numpy as np

    model = build_model(kind, vocab_size=30, seq_len=4, embed_dim=8, units=8)
    out = model(np.zeros((2, 4), dtype="int32"))
    assert out.shape == (2, 30)
    assert np.allclose(np.asarray(out).sum(axis=1), 1.0, atol=1e-4)


def test_build_model_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_model("transformer", 10)


def test_predict_next_returns_k_sorted_suggestions(predictor):
    out = predictor.predict_next("i want to ", k=5)
    assert len(out) == 5
    probs = [s["probability"] for s in out]
    assert probs == sorted(probs, reverse=True)
    assert all(0 < p <= 1 for p in probs)


def test_predict_next_never_suggests_special_tokens(predictor):
    words = {s["word"] for s in predictor.predict_next("i want to ", k=10)}
    assert "<pad>" not in words and "<unk>" not in words


def test_predict_next_learned_the_toy_pattern(predictor):
    # after "i want", the corpus always continues with "to"
    assert predictor.predict_next("i want ", k=3)[0]["word"] == "to"


def test_prefix_completion_only_returns_matching_words(predictor):
    out = predictor.predict_next("i want to g", k=5)
    assert out, "expected at least one completion"
    assert all(s["word"].startswith("g") for s in out)


def test_empty_and_unknown_input_do_not_crash(predictor):
    assert len(predictor.predict_next("", k=5)) == 5
    assert len(predictor.predict_next("zzzz qqqq ", k=3)) == 3


def test_unmatched_prefix_returns_empty_list(predictor):
    assert predictor.predict_next("i want to xyzxyz", k=5) == []


def test_k_is_validated_and_capped(predictor):
    with pytest.raises(ValueError):
        predictor.predict_next("hello ", k=0)
    assert len(predictor.predict_next("hello ", k=10_000)) <= len(predictor.vocab)


def test_missing_files_raise_helpful_error(tmp_path):
    from src.predict import Predictor

    with pytest.raises(FileNotFoundError, match="Train a model first"):
        Predictor.load(tmp_path / "nope.keras", tmp_path / "nope.json")
