import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from text_processor import TextProcessor


def calculate_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    similarity_matrix = cosine_similarity(vectors)
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix


def visualise_similarity_graph(
    similarity_matrix,
    sentences,
    threshold=0.3,
    title="Sentence Similarity Graph",
    save_path=None,
):
    G = nx.Graph()

    for i, _ in enumerate(sentences):
        G.add_node(i, label=f"{i+1}")

    rows, cols = similarity_matrix.shape
    for i in range(rows):
        for j in range(i + 1, cols):
            weight = similarity_matrix[i, j]
            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 10))

    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", alpha=0.8)

    edges = G.edges(data=True)
    weights = [data["weight"] for _, _, data in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)

    labels = {node: G.nodes[node]["label"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")

    plt.show()


def apply_textrank(similarity_matrix: np.ndarray) -> dict:
    graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(graph)

    return scores


def get_relevant_sentences(sentences, scores, num_sentences) -> list:
    ranked_sentences = sorted(
        ((scores[i], i, sentence) for i, sentence in enumerate(sentences)), reverse=True
    )
    relevant_sentences = [
        (i + 1, idx + 1, score, sentence)
        for i, (score, idx, sentence) in enumerate(ranked_sentences[:num_sentences])
    ]

    return relevant_sentences


if __name__ == "__main__":
    file_path = os.path.join("text_rank", "data", "textrank.txt")
    num_relevant_sentences = 5

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        text = text[:100000]

    processor = TextProcessor(text=text, language="german")

    sentences = processor.get_sentences()
    tfidf_matrix = processor.get_tfidf_matrix()
    embeddings = processor.get_embeddings()

    print("Calculating TF-IDF similarity matrix")
    tfidf_similarity_matrix = calculate_similarity_matrix(tfidf_matrix)

    print("Visualizing TF-IDF similarity graph. Close to continue")
    tfidf_graph_path = "text_rank/graphs/tfidf_similarity_graph.png"
    visualise_similarity_graph(
        tfidf_similarity_matrix,
        sentences,
        title="TF-IDF Sentence Similarity Graph",
        save_path=tfidf_graph_path,
    )

    print("Running TextRank on TF-IDF matrix")
    tfidf_scores = apply_textrank(tfidf_similarity_matrix)

    print("Calculating embedding similarity matrix")
    embedding_similarity_matrix = calculate_similarity_matrix(embeddings)

    print("Visualizing embedding similarity graph. Close to continue")
    embedding_graph_path = "text_rank/graphs/embedding_similarity_graph.png"
    visualise_similarity_graph(
        embedding_similarity_matrix,
        sentences,
        title="Sentence Embedding Similarity Graph",
        save_path=embedding_graph_path,
    )

    print("Running TextRank on embedding matrix...")
    embedding_scores = apply_textrank(embedding_similarity_matrix)

    print("\nRESULTS: MOST RELEVANT SENTENCES")

    print(f"\n Top {num_relevant_sentences} sentences based on TF-IDF:")
    tfidf_relevant_sentences = get_relevant_sentences(
        sentences, tfidf_scores, num_relevant_sentences
    )
    for rank, orig_idx, score, sentence in tfidf_relevant_sentences:
        print(f"{rank}. [Sentence {orig_idx}, Score: {score:.4f}] {sentence}")

    print(f"\n Top {num_relevant_sentences} sentences based on Embeddings:")
    embedding_relevant_sentences = get_relevant_sentences(
        sentences, embedding_scores, num_relevant_sentences
    )
    for rank, orig_idx, score, sentence in embedding_relevant_sentences:
        print(f"{rank}. [Sentence {orig_idx}, Score: {score:.4f}] {sentence}")
