from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# https://apiacoa.org/publications/teaching/datasets/google-10000-english.txt

print("Lade Transformer-Modell... (bitte etwas Geduld)")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
print("Transformer erfolgreich geladen!")

def get_embedding(wort):
    inputs = tokenizer(wort, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def wort_rechnung(aufgabe: str):
    try:
        teile = aufgabe.lower().split()
        if len(teile) < 3 or len(teile) % 2 == 0:
            raise ValueError("Ungültiger Ausdruck. Beispiel: 'king - man + woman'")

        vektor = get_embedding(teile[0])

        for i in range(1, len(teile), 2):
            operator, wort = teile[i], teile[i+1]
            emb = get_embedding(wort)
            if operator == "+":
                vektor += emb
            elif operator == "-":
                vektor -= emb
            else:
                raise ValueError(f"Ungültiger Operator: {operator}")

        # Vergleich mit Wortliste (Transformer kennt nicht alle Wörter einzeln)
        with open("words_calculator/common_words.txt") as f:
            testwörter = [line.strip() for line in f if line.strip()]
        ähnlichkeiten = [(w, cosine_similarity(vektor, get_embedding(w))) for w in testwörter]
        ähnlichste = sorted(ähnlichkeiten, key=lambda x: -x[1])[:5]

        print("\nTop 5 ähnliche Wörter:")
        for wort, score in ähnlichste:
            print(f"{wort}: {score:.4f}")
        print(f"\n→ Bestes Ergebnis: {ähnlichste[0][0]}")

    except Exception as e:
        print(f"Fehler: {e}")

wort_rechnung("king - man + woman")
