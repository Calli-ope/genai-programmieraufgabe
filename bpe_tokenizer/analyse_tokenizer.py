from bpe_tokenizer import BPETokenizer
from config import VOCAB_SIZE, TOKENIZERS_DIR, CHARTS_DIR, test_sets
import matplotlib.pyplot as plt
import os

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

def calculate_efficiency(tokenizer, sentences):
    """
    Calculate efficiency metrics for a tokenizer on given sentences.
    
    Args:
        tokenizer: BPETokenizer instance
        sentences: List of sentences to test
        
    Returns:
        Dictionary containing efficiency metrics
    """
    total_tokens = 0
    total_chars = 0
    
    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        total_tokens += len(tokens)
        total_chars += len(sentence)
    
    return {
        'total_tokens': total_tokens,
        'total_chars': total_chars,
        'avg_tokens_per_sentence': total_tokens / len(sentences) if sentences else 0,
        'avg_chars_per_sentence': total_chars / len(sentences) if sentences else 0,
        'avg_tokens_per_char': total_tokens / total_chars if total_chars else 0
    }

def create_tokenizer_charts(german_metrics, english_metrics, combined_metrics, test_sets):
    """Create separate charts for each tokenizer showing average performance on different languages"""
    tokenizers = {
        'German': german_metrics,
        'English': english_metrics, 
        'Combined': combined_metrics
    }
    
    # Set up figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Tokenizer Comparison (Vocab Size: {VOCAB_SIZE})', fontsize=16)
    
    # Create a chart for each tokenizer
    for idx, (tokenizer_name, metrics) in enumerate(tokenizers.items()):
        # Extract average number of tokens for each test language
        language_names = []
        avg_token_counts = []
        
        for test_set_name in test_sets.keys():
            language_names.append(test_set_name)
            avg_token_counts.append(metrics[test_set_name]['avg_tokens_per_sentence'])
        
        # Create the bar chart
        bars = axs[idx].bar(language_names, avg_token_counts)
        axs[idx].set_title(f'{tokenizer_name} Tokenizer')
        axs[idx].set_ylabel('Average Tokens per Sentence')
        axs[idx].set_ylim(0, max(avg_token_counts) * 1.2)
        
        for i, count in enumerate(avg_token_counts):
            axs[idx].text(i, count + 0.5, f"{count:.1f}", ha='center')
    
    plt.tight_layout()
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.savefig(f"{CHARTS_DIR}tokenizer_avg_comparison_vocab{VOCAB_SIZE}.png")
    plt.show()

def main():
    de_tokenizer = BPETokenizer.load(f"{TOKENIZERS_DIR}german_vocab{VOCAB_SIZE}_tokenizer.pkl")
    en_tokenizer = BPETokenizer.load(f"{TOKENIZERS_DIR}english_vocab{VOCAB_SIZE}_tokenizer.pkl")
    combined_tokenizer = BPETokenizer.load(f"{TOKENIZERS_DIR}combined_vocab{VOCAB_SIZE}_tokenizer.pkl")
    
    # Test each tokenizer on all test sets
    tokenizers = {
        'German': de_tokenizer,
        'English': en_tokenizer,
        'Combined': combined_tokenizer
    }
    
    german_metrics = {}
    english_metrics = {}
    combined_metrics = {}
    
    for test_set_name, sentences in test_sets.items():
        print(f"\n=== Testing on {test_set_name} sentences ===")
        
        for tokenizer_name, tokenizer in tokenizers.items():
            print(f"\n--- Using {tokenizer_name} Tokenizer ---")
            test_tokenizer(tokenizer, sentences)
            if tokenizer_name == 'German':
                german_metrics[test_set_name] = calculate_efficiency(tokenizer, sentences)
            elif tokenizer_name == 'English':
                english_metrics[test_set_name] = calculate_efficiency(tokenizer, sentences)
            else:
                combined_metrics[test_set_name] = calculate_efficiency(tokenizer, sentences)
    
    create_tokenizer_charts(german_metrics, english_metrics, combined_metrics, test_sets)
    
    # Print out which tokenizer is most efficient for each dataset
    for test_set_name in test_sets.keys():
        metrics = [
            ("German", german_metrics[test_set_name]['avg_tokens_per_sentence']),
            ("English", english_metrics[test_set_name]['avg_tokens_per_sentence']),
            ("Combined", combined_metrics[test_set_name]['avg_tokens_per_sentence'])
        ]
        
        most_efficient = min(metrics, key=lambda x: x[1])
        print(f"\nFor {test_set_name} sentences, the {most_efficient[0]} tokenizer is most efficient")
        print(f"with an average of {most_efficient[1]:.2f} tokens per sentence.")

if __name__ == "__main__":
    main()