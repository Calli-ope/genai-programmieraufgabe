# Configuration parameters for BPE tokenizer

# Vocabulary size for tokenizers
TRAIN_VOCAB_SIZE = 1500 # Used when training tokenizers
TEST_VOCAB_SIZE = 1500   # Used when loading tokenizers for testing

test_sets = {
    'German': [
        "Die Künstliche Intelligenz hat bedeutende Fortschritte in der Sprachverarbeitung gemacht.",
        "Die Programmierung von Sprachmodellen erfordert ein Verständnis von Tokenisierungsalgorithmen.",
        "BPE-Tokenisierung funktioniert durch iteratives Zusammenführen häufiger Zeichenpaare."
    ],
    'English': [
        "Artificial Intelligence has made significant progress in language processing.",
        "Programming natural language models requires understanding of tokenization algorithms.",
        "BPE tokenization works by merging common character pairs iteratively."
    ],
    'Mixed': [
        "Die Artificial Intelligence hat significant progress in der language processing gemacht.",
        "Programming von Sprachmodellen erfordert ein understanding von tokenization algorithms.",
        "BPE-Tokenisierung works by merging häufiger character pairs iteratively."
    ]
}

# Directory paths for storing outputs
CHARTS_DIR = "bpe_tokenizer/charts/"
TOKENIZERS_DIR = "bpe_tokenizer/trained_tokenizers/"

# Paths to training data
GERMAN_CORPUS_PATH = "bpe_tokenizer/data/de.txt"
ENGLISH_CORPUS_PATH = "bpe_tokenizer/data/en.txt"