import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sentence_transformers import SentenceTransformer
import re

nltk.download('stopwords', quiet=True)

class TextProcessor:
    """
    Klasse zur Zerlegung von Texten in Sätze und Erzeugung verschiedener Satzrepräsentationen.
    """
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Languague model is loading...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
    def split_into_sentences(self, text):
        """Text in Sätze zerlegen mit Spacy"""
        text = re.sub(r'\s+', ' ', text)  # Mehrfache Leerzeichen entfernen
        
        # Mit Spacy in Sätze zerlegen
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Leere Sätze entfernen
        sentences = [s for s in sentences if s.strip()]
        
        print(f"Text in {len(sentences)} Sätze zerlegt.")
        return sentences
    
    def create_tfidf_vectors(self, sentences):
        """TF-IDF-Vektoren für Sätze berechnen"""
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        print(f"TF-IDF-Vektoren erstellt: Form {tfidf_matrix.shape}")
        return tfidf_matrix
    
    def create_embeddings(self, sentences):
        """Satzembeddings mit Sentence-Transformer erstellen"""
        embeddings = self.sentence_transformer.encode(sentences)
        
        print(f"Sentence-Transformer-Embeddings erstellt: Form {embeddings.shape}")
        return embeddings
