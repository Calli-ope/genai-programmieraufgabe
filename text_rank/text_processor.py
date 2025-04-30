import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

nltk.download("punkt")
nltk.download("punkt_tab")


class TextProcessor:
    def __init__(
        self,
        text,
        language="german",
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.text = text
        self.language = language
        self.embedding_model_name = embedding_model
        self.sentences = []
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.embeddings = None

        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        print(
            f"TextProcessor initialised for language: {language}, "
            f"Embedding-Modell: {embedding_model}"
        )

        self.split_into_sentences()
        self.generate_tfidf_vectors()
        self.generate_embeddings()

    def split_into_sentences(self):
        self.sentences = nltk.sent_tokenize(self.text, language=self.language)
        if not self.sentences:
            raise ValueError("Could not split text into sentences.")
        print(f"Found {len(self.sentences)} sentences.")

    def generate_tfidf_vectors(self):
        print("\nGenerating TF-IDF vectors")
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sentences)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def generate_embeddings(self):
        print(
            f"\nGenerating sentence embeddings using model: '{self.embedding_model_name}'"
        )
        model = SentenceTransformer(self.embedding_model_name)
        self.embeddings = model.encode(self.sentences)
        print(f"Embeddings matrix shape: {self.embeddings.shape}")

    def get_sentences(self) -> list[str]:
        return self.sentences

    def get_tfidf_matrix(self) -> np.ndarray:
        return self.tfidf_matrix.toarray()

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings
