#!/usr/bin/env python3
"""
 Placement holder
"""

# ----------------- Futures -----------------
from __future__ import annotations # import future annotation for type hinting

# ----------------- Standard Library -----------------
from pathlib import Path #import pathlib for resolving path to the Corpus file
import re # import regex


# ----------------- Third Party Library -----------------


# ----------------- Application Imports -----------------


# ----------------- Module-level Configuration -----------------

def repo_root(root_path: Path):
    """
    place holder docstring
    """
    # For loop that walks up the directory tree from the root_path, checking for pyproject.toml and .git files to ID the repo root.
    for path in [root_path] + list(root_path.parents):
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path
    return Path.cwd() # Fallback to current working directory if the root is not found

ROOT = repo_root(Path(__file__).resolve()) # Dunder method to resolve the path to the current file and find the repo root

Corpus_Path = ROOT / "src" / "metal" / "corpus" / "war_and_peace.txt" # Actual path to the corpus file, using pathlib to construct the path in a platform-independent way.

with open(Corpus_Path, "r", encoding="utf-8-sig") as f: # with open always ensures closing, utf-8 ensures proper encoding -sig ensures /ufeff is not printed at
    # the beginnng of the file.
    CORPUS = f.read() # read the corpus in entirety at one time.

clean_corpus = re.split(r'([,.?_!"()\']|--|\s)', CORPUS) # clean the corpus with regex
corpus_tokens = [item.strip() for item in clean_corpus if item.strip()] # list comprehension to strip whitespace and filter empty tokens.
# Stripping empty token can actually be to detriment but this will be fixed later. This is simply a "check"

war_and_peace = sorted(set(corpus_tokens)) # Create a sorted set of unique words and tokens.
war_and_peace.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(war_and_peace) # 23,562 unique tokens
# create the vocabulary
vocab = {token: i for i, token in enumerate(war_and_peace)}
print(vocab_size) # confirms the size of 23,562 unique tokens.


if __name__ == "__main__":
    print(CORPUS[1:30]) # print the first 30 characters
    print(clean_corpus[1:51]) # print the first 50 lines of the cleaned corpus
    print(corpus_tokens[:30])
    print(vocab_size) # confirms the size of 23,562 unique tokens.
    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)
