"""Text cleaning, tokenizing and vocabulary handling.

Pipeline (matches the design sketch):  raw text -> clean -> tokenize -> token ids
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import config

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
# Every newline is a hard boundary: in the SwiftKey corpus each line is an independent
# tweet / paragraph, so words from neighbouring lines must never form a training window.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ELONGATED_RE = re.compile(r"(.)\1{2,}")  # "sooooo" -> "soo"
_QUOTE_MAP = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'})
_GUTENBERG_START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_GUTENBERG_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)

# social-media noise that plain literary text never has
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_RETWEET_RE = re.compile(r"^\s*rt\s+(@\w+\s*:?\s*)?", re.I)
_HASHTAG_RE = re.compile(r"#(\w+)")  # keep the word, drop the '#'


def strip_social_noise(text: str) -> str:
    """Remove URLs and @mentions, unwrap hashtags, and drop a leading 'RT @user:'.

    Only relevant for Twitter-sourced text; harmless no-op on prose.
    """
    text = _RETWEET_RE.sub("", text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    return text


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg licence header and footer, if present."""
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def strip_accents(text: str) -> str:
    """Fold accented letters to ASCII ("café" -> "cafe") so they are not split into fragments."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    """Strip social noise, fold accents, lowercase, normalise quotes, squash "sooooo", collapse whitespace."""
    text = strip_social_noise(text)
    text = strip_accents(text.translate(_QUOTE_MAP)).lower()
    text = _ELONGATED_RE.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens. Keeps inner apostrophes (don't)."""
    return _TOKEN_RE.findall(clean_text(text))


def split_sentences(text: str, min_tokens: int = 2) -> list[list[str]]:
    """Turn a long text into a list of token lists, one per sentence (or per line)."""
    sentences = []
    for chunk in _SENTENCE_SPLIT_RE.split(strip_gutenberg(text)):
        tokens = tokenize(chunk)
        if len(tokens) >= min_tokens:
            sentences.append(tokens)
    return sentences


class Vocabulary:
    """Word <-> id mapping with <pad>=0 and <unk>=1 reserved."""

    def __init__(self, words: Iterable[str]):
        self.itos: list[str] = [config.PAD_TOKEN, config.UNK_TOKEN]
        self.itos += [w for w in words if w not in (config.PAD_TOKEN, config.UNK_TOKEN)]
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(
        cls,
        sentences: Iterable[list[str]],
        max_size: int = config.MAX_VOCAB,
        min_freq: int = config.MIN_FREQ,
    ) -> "Vocabulary":
        """Build from training sentences only, most frequent words first."""
        counts = Counter(tok for sent in sentences for tok in sent)
        kept = [w for w, c in counts.most_common() if c >= min_freq]
        return cls(kept[: max_size - 2])

    def encode(self, tokens: Iterable[str]) -> list[int]:
        unk = config.UNK_ID
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls([])
        vocab.itos = data["itos"]
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
        return vocab
