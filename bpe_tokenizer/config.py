VOCAB_SIZE = 1500

test_sets = {
    "German": [
        "Die Programmierung von Sprachmodellen erfordert ein Verständnis von Tokenisierungsalgorithmen.",
        "Das Wetter ist heute wirklich schön.",
        "Kannst du mir bitte den Weg zum nächsten Bahnhof zeigen?",
    ],
    "English": [
        "Programming natural language models requires understanding of tokenization algorithms.",
        "The weather is really nice today.",
        "Could you please show me the way to the nearest train station?",
    ],
    "Mixed": [
        "Programming von Sprachmodellen erfordert ein understanding von tokenization algorithms.",
        "Das weather is really schön today.",
        "Could you bitte show me den Weg zum nearest Bahnhof?",
    ],
}

CHARTS_DIR = "bpe_tokenizer/charts/"
TOKENIZERS_DIR = "bpe_tokenizer/trained_tokenizers/"

GERMAN_CORPUS_PATH = "bpe_tokenizer/data/de.txt"
ENGLISH_CORPUS_PATH = "bpe_tokenizer/data/en.txt"
