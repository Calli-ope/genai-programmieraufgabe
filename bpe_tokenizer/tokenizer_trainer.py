from bpe_tokenizer import BPETokenizer
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
    
    os.makedirs("bpe_tokenizer/trained_tokenizers", exist_ok=True)
    tokenizer.save(f"bpe_tokenizer/trained_tokenizers/{name}_tokenizer.pkl")
    
    return tokenizer

def test_tokenizer(tokenizer, sentences):
    """
    Test the tokenizer on new sentences.
    
    Args:
        tokenizer: BPETokenizer instance
        sentences: List of sentences to test
    """
    for sentence in sentences:
        print(f"\nOriginal: {sentence}")
        tokens = tokenizer.tokenize(sentence)
        print(f"Tokens: {tokens}")
        ids = tokenizer.encode(sentence)
        print(f"Encoded: {ids}")

def main():
    vocab_size = 1000 # low value -> more and shorter tokens, high value -> less and longer tokens
    de_tokenizer = train_tokenizer(["bpe_tokenizer/data/de.txt"], vocab_size, "german")
    en_tokenizer = train_tokenizer(["bpe_tokenizer/data/en.txt"], vocab_size, "english")
    combined_tokenizer = train_tokenizer(["bpe_tokenizer/data/de.txt", "bpe_tokenizer/data/en.txt"], vocab_size, "combined")
    
    print("Loading trained tokenizers...")
    de_tokenizer = BPETokenizer.load("bpe_tokenizer/trained_tokenizers/german_tokenizer.pkl")
    en_tokenizer = BPETokenizer.load("bpe_tokenizer/trained_tokenizers/english_tokenizer.pkl")
    combined_tokenizer = BPETokenizer.load("bpe_tokenizer/trained_tokenizers/combined_tokenizer.pkl")
    
    german_sentences = [
        "Hallo wie geht es dir heute?",
        "Ich lerne maschinelles Lernen.",
        "Donaudampfschifffahrtsgesellschaftskapitän ist ein langes Wort.",  # Long compound word
        "Überraschung! Die Äpfel und Birnen sind über die Straße.",  # Umlauts
        "Stiftung Warentest hat dieses Produkt mit gut bewertet."  # German institution
    ]
    
    english_sentences = [
        "Hello how are you today?",
        "I am learning machine learning.",
        "Don't worry, I can't find my phone either.",  # Contractions
        "The quick brown fox jumps over the lazy dog.",  # Common English pangram
        "Machine learning algorithms need a lot of data."  # Technical terminology
    ]
    
    mixed_sentences = [
        "Ich nutze machine learning für meine Hausarbeit.",  # German with English term
        "The Kindergarten teacher speaks Deutsch very well.",  # English with German words
        "Das ist ein interessantes Framework für Deep Learning.",  # Code-switching
        "COVID-19 pandemic affected both Deutschland and England."  # Names and numbers
    ]
    
    print("\n=== Testing German Tokenizer ===")
    test_tokenizer(de_tokenizer, german_sentences)
    
    print("\n=== Testing English Tokenizer ===")
    test_tokenizer(en_tokenizer, english_sentences)
    
    print("\n=== Testing Combined Tokenizer with Mixed Sentences ===")
    test_tokenizer(combined_tokenizer, mixed_sentences)

if __name__ == "__main__":
    main()