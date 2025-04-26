# Configuration parameters for BPE tokenizer

# Vocabulary size for tokenizers
VOCAB_SIZE = 1500 # Adjust for training and testing

test_sets = {
    'German': [
        "Die Programmierung von Sprachmodellen erfordert ein Verständnis von Tokenisierungsalgorithmen.",
        "Das Wetter ist heute wirklich schön.",
        "Kannst du mir bitte den Weg zum nächsten Bahnhof zeigen?"
    ],
    'English': [
        "Programming natural language models requires understanding of tokenization algorithms.",
        "The weather is really nice today.",
        "Could you please show me the way to the nearest train station?"
    ],
    'Mixed': [
        "Programming von Sprachmodellen erfordert ein understanding von tokenization algorithms.",
        "Das weather is really schön today.",
        "Could you bitte show me den Weg zum nearest Bahnhof?"
    ]
}

# Directory paths for storing outputs
CHARTS_DIR = "bpe_tokenizer/charts/"
TOKENIZERS_DIR = "bpe_tokenizer/trained_tokenizers/"

# Paths to training data
GERMAN_CORPUS_PATH = "data/de.txt"
ENGLISH_CORPUS_PATH = "data/en.txt"