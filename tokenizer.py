#!/usr/bin/env python3
"""
 SimpleTokenizer Class for the War and Peace corpus.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
import re

# ----------------- Third Party Library -----------------


# ----------------- Application Imports -----------------
from lab import vocab

# ----------------- Module-level Configuration -----------------


class SimpleTokenizer:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # use our str_to_int to turn from string to integer and create ids
        ids = [self.str_to_int[s] for s in preprocessed] 
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids]) # join on spaces
        text = re.sub(r'([,.?_!"()\']|--|\s)', r'\1', text) # regex pattern
        return text

tokenizer = SimpleTokenizer(vocab)
text = """This eBook is for the use of anyone anywhere in the United States and"""
ids = tokenizer.encode(text)

if __name__ == "__main__":

    print(ids)
    print(tokenizer.decode(ids))

