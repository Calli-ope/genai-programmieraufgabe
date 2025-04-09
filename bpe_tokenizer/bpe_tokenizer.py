import re
import collections
import pickle
from typing import List

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.pattern = None
    
    def train(self, texts: List[str]):
        """Train the BPE tokenizer on the given corpus."""
        # Initialize with character-level tokens
        word_counts = collections.Counter()
        for text in texts:
            text = re.sub(r'\s+', ' ', text)
            words = text.split()
            word_counts.update(words)
        
        char_vocab = set()
        for word, _ in word_counts.items():
            for char in word:
                char_vocab.add(char)

        special_tokens = ['<unk>', '<pad>', '<s>', '</s>']
        char_vocab.update(special_tokens)

        # Initialize the vocabulary with character-level tokens
        self.vocab = {char: i for i, char in enumerate(char_vocab)}

        # Create initial word splits
        splits = {}
        for word, count in word_counts.items():
            splits[word] = [char for char in word]
        
        # Merge most frequent pairs until vocabulary size is reached
        merges = {}
        vocab_size = len(self.vocab)

        while vocab_size < self.vocab_size:
            # Count frequency of adjacent pairs
            pairs = collections.Counter()
            for word, freq in word_counts.items():
                word_pieces = splits[word]
                if len(word_pieces) == 1:
                    continue

                for i in range(len(word_pieces) - 1):
                    pair = (word_pieces[i], word_pieces[i + 1])
                    pairs[pair] += freq

            if not pairs:  # No more pairs to merge
                break

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            best_pair_str = ''.join(best_pair)

            # Add to vocab and merges
            self.vocab[best_pair_str] = vocab_size
            merges[best_pair] = best_pair_str
            vocab_size += 1

            # Apply the merge to all words
            for word in word_counts:
                pieces = splits[word]
                i = 0
                while i < len(pieces) - 1:
                    current_pair = (pieces[i], pieces[i + 1])
                    if current_pair == best_pair:
                        pieces[i] = best_pair_str
                        del pieces[i + 1]
                    else:
                        i += 1
                splits[word] = pieces

        self.merges = merges

        # Create a regex pattern for tokenization
        self.pattern = re.compile(r'\s+|' + '|'.join(re.escape(k) for k in sorted(self.vocab.keys(), key=len, reverse=True)))

        return self

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text using the trained BPE tokenizer."""
        if self.pattern is None:
            raise ValueError("Tokenizer has not been trained yet.")

        if self.pattern:
            tokens = [match.group(0) for match in self.pattern.finditer(text)]
            return tokens
        else:
            return [char for char in text]
        
    def encode(self, text: str) -> List[int]:
        """Encode the input text into a list of token IDs."""
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['<unk>'])
        return ids

    def save(self, path: str):
        """"Save the tokenizer to a file."""
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': self.merges
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        """"Load the tokenizer to a file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.merges = data['merges']
        tokenizer.pattern = re.compile(r'\s+|' + '|'.join(re.escape(k) for k in sorted(tokenizer.vocab.keys(), key=len, reverse=True)))

        return tokenizer