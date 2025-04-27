import numpy as np

print("Lade GloVe-Modell... (bitte etwas Geduld)")
def lade_glove_modell(pfad):
    modell = {}
    with open(pfad, 'r', encoding='utf-8') as f:
        for zeile in f:
            teile = zeile.strip().split()
            wort = teile[0]
            vektor = np.array(teile[1:], dtype=np.float32)
            modell[wort] = vektor
    return modell

glove = lade_glove_modell("words_calculator/glove.6B.300d.txt")  # Pfad anpassen
print("GloVe-Modell erfolgreich geladen!")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def wort_rechnung(aufgabe: str):
    try:
        teile = aufgabe.lower().split()
        if len(teile) < 3 or len(teile) % 2 == 0:
            raise ValueError("Ungültiger Ausdruck. Beispiel: 'king - man + woman'")

        base = teile[0]
        if base not in glove:
            print(f"Wort nicht im Vokabular: '{base}'")
            return
        vektor = glove[base]

        for i in range(1, len(teile), 2):
            operator, wort = teile[i], teile[i+1]
            if wort not in glove:
                print(f"Wort nicht im Vokabular: '{wort}'")
                return
            if operator == "+":
                vektor += glove[wort]
            elif operator == "-":
                vektor -= glove[wort]
            else:
                raise ValueError(f"Ungültiger Operator: {operator}")

        # Ähnlichstes Wort finden
        beste_wörter = sorted(
            ((w, cosine_similarity(vektor, v)) for w, v in glove.items() if w not in teile),
            key=lambda x: -x[1]
        )[:5]

        print("\nTop 5 ähnliche Wörter:")
        for wort, score in beste_wörter:
            print(f"{wort}: {score:.4f}")
        print(f"\n→ Bestes Ergebnis: {beste_wörter[0][0]}")

    except Exception as e:
        print(f"Fehler: {e}")

wort_rechnung("king - man + woman")
wort_rechnung("paris - france + germany")
