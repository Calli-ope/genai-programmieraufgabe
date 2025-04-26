from bpe_tokenizer import BPETokenizer
from config import VOCAB_SIZE, GERMAN_CORPUS_PATH, ENGLISH_CORPUS_PATH, TOKENIZERS_DIR
import os

def load_text_data(file_path):
    """Load text data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def train_tokenizer(dataset_paths, vocab_size, name):
    """
    Train tokenizer on the given dataset.
    
    Args:
        dataset_paths: List of text file paths
        vocab_size: Vocabulary size
        name: Name for saving the tokenizer
    """
    all_texts = []
    for path in dataset_paths:
        texts = load_text_data(path)
        all_texts.extend(texts)
    
    print(f"Training {name} tokenizer on {len(all_texts)} texts with vocab size {vocab_size}...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(all_texts)
    
    os.makedirs(TOKENIZERS_DIR, exist_ok=True)
    tokenizer.save(f"{TOKENIZERS_DIR}{name}_vocab{vocab_size}_tokenizer.pkl")
    
    return tokenizer

def main():
    train_tokenizer([GERMAN_CORPUS_PATH], VOCAB_SIZE, "german")
    train_tokenizer([ENGLISH_CORPUS_PATH], VOCAB_SIZE, "english")
    train_tokenizer([GERMAN_CORPUS_PATH, ENGLISH_CORPUS_PATH], VOCAB_SIZE, "combined")
        
if __name__ == "__main__":
    main()