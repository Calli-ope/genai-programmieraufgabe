import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from text_processor import TextProcessor


class SimilarityGraph:
    """
    Klasse zur Erstellung eines Ähnlichkeitsgraphen zwischen Sätzen und Anwendung des TextRank-Algorithmus.
    """
    def build_similarity_graph(self, matrix, threshold=0.3):
        """
        Ähnlichkeitsgraph basierend auf Kosinus-Ähnlichkeit erstellen
        
        Parameters:
        - matrix: Entweder TF-IDF-Matrix oder Embedding-Matrix
        - threshold: Schwellenwert für die Ähnlichkeit (Kanten werden nur erstellt, 
                    wenn die Ähnlichkeit über diesem Wert liegt)
        
        Returns:
        - graph: NetworkX-Graph-Objekt
        - similarity_matrix: Matrix mit Ähnlichkeitswerten zwischen allen Satzpaaren
        """
        # Kosinus-Ähnlichkeit berechnen
        # Bei Sparse-Matrix (TF-IDF) oder Dense-Matrix (Embeddings)
        if hasattr(matrix, 'toarray'):
            similarity_matrix = cosine_similarity(matrix)
        else:
            similarity_matrix = cosine_similarity(matrix)
        
        # Graph erstellen
        graph = nx.Graph()
        for i in range(len(similarity_matrix)):
            graph.add_node(i)
        
        # Kanten basierend auf Ähnlichkeitswerten hinzufügen
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix)):
                sim = similarity_matrix[i][j]
                if sim > threshold:
                    graph.add_edge(i, j, weight=sim)
        
        print(f"Graph erstellt mit {graph.number_of_nodes()} Knoten und {graph.number_of_edges()} Kanten")
        print(f"Graphdichte: {nx.density(graph):.4f}")
        
        return graph, similarity_matrix
    
    def apply_textrank(self, graph):
        """
        TextRank-Algorithmus auf den Graphen anwenden
        
        Parameters:
        - graph: NetworkX-Graph-Objekt
        
        Returns:
        - scores: Dictionary mit TextRank-Scores für jeden Knoten (Satz)
        """
        # PageRank-Algorithmus anwenden (Basis für TextRank)
        scores = nx.pagerank(graph)
        
        # Top-5 Knoten nach Score ausgeben
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop-5 Sätze nach TextRank-Score:")
        for node, score in top_nodes:
            print(f"Satz {node}: Score {score:.4f}")
        
        return scores
    
    def get_top_sentences(self, sentences, scores, top_n=5):
        """
        Die Top-N-Sätze basierend auf TextRank-Scores extrahieren
        
        Parameters:
        - sentences: Liste aller Sätze
        - scores: Dictionary mit TextRank-Scores
        - top_n: Anzahl der zu extrahierenden Top-Sätze
        
        Returns:
        - top_sentences: Liste von Tupeln (Index, Satz, Score)
        """
        # Sätze nach Score sortieren
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), 
                                 reverse=True)
        
        # Top-N Sätze auswählen und in ursprünglicher Reihenfolge zurückgeben
        selected_indices = [ranked_sentences[i][1] for i in range(min(top_n, len(ranked_sentences)))]
        selected_indices.sort()
        
        top_sentences = [(i, sentences[i], scores[i]) for i in selected_indices]
        
        return top_sentences
    
    def visualize_graph(self, graph, scores=None, title="Satzähnlichkeitsgraph"):
        """
        Graph visualisieren mit optionaler Knotengröße basierend auf Scores
        
        Parameters:
        - graph: NetworkX-Graph-Objekt
        - scores: Dictionary mit Scores für jeden Knoten (optional)
        - title: Titel für die Grafik
        
        Returns:
        - plt: Matplotlib-Figure-Objekt
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph, seed=42)
        
        if scores:
            # Knotengröße proportional zum Score
            node_sizes = [scores[node] * 3000 for node in graph.nodes()]
        else:
            node_sizes = 300
            
        # Kanten mit ihrer Gewichtung zeichnen
        edge_weights = [graph[u][v]['weight'] * 2 for u, v in graph.edges()]
        
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, alpha=0.7, 
                               node_color='lightblue')
        nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(graph, pos, font_size=10)
        
        plt.title(title)
        plt.axis('off')
        return plt
    
    def visualize_similarity_matrix(self, similarity_matrix, title="Ähnlichkeitsmatrix"):
        """
        Visualisierung der Ähnlichkeitsmatrix als Heatmap
        
        Parameters:
        - similarity_matrix: Matrix mit Ähnlichkeitswerten
        - title: Titel für die Grafik
        
        Returns:
        - plt: Matplotlib-Figure-Objekt
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=False, cmap='viridis')
        plt.title(title)
        return plt

if __name__ == "__main__":
    with open('data/en.txt', 'r', encoding='utf-8') as f:
        demo_text = f.read()
        demo_text = demo_text[:100000]
    
    processor = TextProcessor()
    sentences = processor.split_into_sentences(demo_text)
    tfidf_matrix = processor.create_tfidf_vectors(sentences)
    embeddings = processor.create_embeddings(sentences)
    
    graph_builder = SimilarityGraph()
    
    print("\n--- TF-IDF-basierter Graph ---")
    tfidf_graph, tfidf_sim_matrix = graph_builder.build_similarity_graph(tfidf_matrix)
    tfidf_scores = graph_builder.apply_textrank(tfidf_graph)
    
    print("\n--- Embedding-basierter Graph ---")
    emb_graph, emb_sim_matrix = graph_builder.build_similarity_graph(embeddings)
    emb_scores = graph_builder.apply_textrank(emb_graph)
    
    print("\nTop-3 Sätze mit TF-IDF:")
    tfidf_top = graph_builder.get_top_sentences(sentences, tfidf_scores, top_n=3)
    for idx, sentence, score in tfidf_top:
        print(f"[{idx}] ({score:.4f}): {sentence}")
    
    print("\nTop-3 Sätze mit Embeddings:")
    emb_top = graph_builder.get_top_sentences(sentences, emb_scores, top_n=3)
    for idx, sentence, score in emb_top:
        print(f"[{idx}] ({score:.4f}): {sentence}")
    
    graph_builder.visualize_graph(tfidf_graph, tfidf_scores, 
                                "TF-IDF Satzähnlichkeitsgraph").savefig("text_rank/graphs/tfidf_graph.png")
    graph_builder.visualize_graph(emb_graph, emb_scores, 
                                "Embedding Satzähnlichkeitsgraph").savefig("text_rank/graphs/embedding_graph.png")
