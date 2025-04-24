import gensim.downloader as api

print("Lade Modell... (bitte etwas Geduld)")
model = api.load("word2vec-google-news-300")
print("Modell erfolgreich geladen!")

def wort_rechnung(aufgabe: str):
    try:
        teile = aufgabe.lower().split()
        if len(teile) < 3 or len(teile) % 2 == 0:
            raise ValueError("Ungültiger Ausdruck. Beispiel: 'king - man + woman'")

        positive = [teile[0]]
        negative = []

        for i in range(1, len(teile), 2):
            operator, wort = teile[i], teile[i+1]
            if wort not in model:
                print(f"Wort nicht im Vokabular: '{wort}'")
                return
            if operator == "+":
                positive.append(wort)
            elif operator == "-":
                negative.append(wort)
            else:
                raise ValueError(f"Ungültiger Operator: {operator}")

        ähnlichste = model.most_similar(positive=positive, negative=negative, topn=5)

        print("\nTop 5 ähnliche Wörter:")
        for wort, score in ähnlichste:
            print(f"{wort}: {score:.4f}")

        print(f"\n→ Bestes Ergebnis: {ähnlichste[0][0]}")

    except Exception as e:
        print(f"Fehler: {e}")

wort_rechnung("king - man + woman")
