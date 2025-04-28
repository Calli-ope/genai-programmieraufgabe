# Programmieraufgaben - Seminar zu Generativer KI

Konrad Christoph Martens; Finnian Kühn

## 01 - BPE-Tokenizer

### a)

1. In der [Konfigurationsdatei](bpe_tokenizer/config.py) gewünschte Vokabulargröße setzen
2. [train_tokenizer.py](bpe_tokenizer/train_tokenizer.py) ausführen
3. [analyse_tokenizer.py](bpe_tokenizer/analyse_tokenizer.py) ausführen

### b)

Für den Vergleich wurden drei Tokenizer (Deutsch, Englisch, Deutsch + Englisch) auf dem verlinkten Datensatz des Auswärtigen Amtes trainiert. Dabei wurden drei Versionen mit unterschiedlicher Vokabulargröße (500, 1000, 1500) trainiert.

Im Rahmen der Analyse wurden alle Versionen der Tokenizer mit jeweils drei deutschen, englischen und gemischten Sätzen getestet, die jeweils die gleiche Aussage enthalten und in der [Konfigurationsdatei](bpe_tokenizer/config.py) zu finden sind. Für die Charts wurde die durchschnittliche Anzahl der Tokens pro Satz berechnet, die auf der y-Achse gezeigt wird. Für jede Vokabulargröße werden alle drei Tokenizer für alle drei Sprachen verglichen.

**Vokabulargröße: 500**

![](bpe_tokenizer/charts/tokenizer_avg_comparison_vocab500.png)

**Vokabulargröße: 1000**

![](bpe_tokenizer/charts/tokenizer_avg_comparison_vocab1000.png)

**Vokabulargröße: 1500**

![](bpe_tokenizer/charts/tokenizer_avg_comparison_vocab1500.png)

**Ergebnisse**

Auffällig ist, dass die durchschnittliche Anzahl der Tokens pro Satz mit zunehmender Vokabulargröße abnimmt. Außerdem erzielen die auf Deutsch bzw. Englisch trainierten Tokenizer die besten Ergebnisse für Sätze in der jeweiligen Sprache. Die Anzahl der Token für gemischte Sätze liegt für beide Tokenizer zwischen den Ergebnissen für deutsche und englische Sätze. Darüber hinaus kann der kombinierte Tokenizer englischen Sätze besser kodieren als gemischte Sätze. Insgesamt können englische Sätze über alle Vokabulargrößen hinweg mit durchschnittlich am wenigsten Tokens kodiert werden, während deutsche und gemischte Sätze vergleichbar viele Tokens benötigen.

### c)

Die Tokenanzahl pro Satz hängt direkt von der Vokabulargröße ab. Kleine Vokabulare (ca. 500 Einheiten) erzeugen mehr kurze Tokens, große Vokabulare (ca. 1500) weniger und längere Tokens, die ganze Wörter oder bedeutungsvolle Wortteile umfassen können. Dabei gilt: Kleine Vokabulare ermöglichen schnelleres Training bei ineffizientem Encoding, große Vokabulare verlangsamen das Training, bieten aber effizienteres Encoding. Die größten Effizienzgewinne liegen zwischen 500 und 1000 Vokabeleinheiten.

Die Effizienzunterschiede zwischen den Tokenizern spiegeln die sprachspezifischen Eigenschaften wider. Der deutsche Tokenizer hat beispielsweise gelernt, deutsche Komposita und morphologische Strukturen effizient zu kodieren, während der englische Tokenizer die häufigen englischen Wortbausteine besser repräsentiert. Beide zeigen deutliche Schwächen bei fremdsprachigen Texten.
Gemischte Sätze liegen in ihrer Tokenanzahl zwischen den Werten der spezialisierten Modelle. Interessanterweise erweist sich der kombinierter, multilingualer Tokenizer sowohl für gemischte Sätze als vorteilhaft, da er beide Sprachmuster abdeckt, als auch für englische Sätze, da diese mit nochmal weniger Token als die gemischten Sätze abgedeckt werden.

Die Ergebnisse zeigen auch, dass eine größere Vokabulargröße zu einer kompakteren Darstellung von Informationen führt, da mehr bedeutungsvolle Subword-Einheiten in einem einzelnen Token erfasst werden können.

Der kombinierte deutsch-englische Tokenizer zeigt ausgewogene Leistung in beiden Sprachen. Er erreicht zwar nicht die Spitzeneffizienz der spezialisierten Modelle, benötigt aber weniger Tokens für fremdsprachliche Sätze als die jeweils unpassenden Einzelsprach-Tokenizer.

Die optimale Wahl hängt vom Anwendungsszenario ab:

-   Einsprachige Systeme: Sprachspezifische Tokenizer mit großem Vokabular
-   Mehrsprachige/gemischte Inhalte: Kombinierte Tokenizer für konsistente Ergebnisse über Sprachgrenzen hinweg
-   Ressourcenbeschränkte Szenarien: Multilingualer Ansatz für beste Balance zwischen Kompaktheit und Flexibilität

## 02 - Wort-Taschenrechner

### a)

Um den Wortrechner umzusetzen, wurde gensim verwendet. Darüber können sehr einfach Modelle heruntergeladen werden und genutzt werden, um Embeddings zu erhalten und mit ihnen zu rechnen. Zunächst wurde ein Word2Vec Modell verwendet, was zu einem späteren Zeitpunkt durch "glove-wiki-gigaword-50" ersetzt wurde, um Aufgabenteil b) zu entsprechen ([Quelle](https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html)).

1. Eingabe überprüfen und in einzelne Elemente aufteilen
2. Operator identifizieren und anhand dieses, Wörter in zwei Listen (positiv und negativ) aufteilen
3. Funktion auf Modell aufrufen und Listen übergeben -> Rückgabewert ist Liste aus Wörtern und Übereinstimmungsgrad (0-1)
4. Die 5 besten Ergebnisse ausgeben

An das Modell übergebene Wörter müssen aus Ergebnisses gefiltert werden, da diese sonst häufig die größte Übereinstimmung haben ([Quelle](https://blog.esciencecenter.nl/king-man-woman-king-9a7fd2935a85)). Bei gensim geschieht das bereits automatisch

#### Weitere Beispiele und erwartetes ergebnis:

-   winter - cold + hot = summer
-   pizza - italy + japan = sushi
-   paris - france + germany = berlin
-   paris - france + italy = rome
-   teacher - school + court = lawyer
-   museum - art + science = labroratory (oder ähnlich)
-   uncle - man + woman = aunt
-   apple - iphone + galaxy = samsung (Hier fallen unterschiedliche Bedeutungen für das gleiche Wort stark ins Gewicht)

### b)

#### Implementierung mit BERT

Für die Umsetzung mit Transfomer-Technologie wird die Bibliothek "transformers" verwendet, über die vortrainierte Modelle und Tokenizer (in diesem Fall BERT) genutzt werden können [Quelle](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#2-input-formatting). Diese werden genutzt, um zunächst die Embeddings von 10000 englischen Wörtern aus einer Text-Datei zu berechnen, damit dies nicht bei jedem Funktionsaufruf getan werden muss. Da es sich lediglich um einzelne Wörter ohne Kontext handelt, wird das Ergebnis jedes mal das gleiche sein, weshalb dieser Schritt einmalig am Anfang ausgeführt werden kann:

1. Wörter auslesen und in Liste speichern
2. Für jedes Wort:

    1. Tokenisieren und dabei Sondertokens hinzufügen
    2. Modell mit input aufrufen
    3. [CLS-Token](https://aditya007.medium.com/understanding-the-cls-token-in-bert-a-comprehensive-guide-a62b3b94a941) zur weiteren Berechnung verwenden, da dieses die "Zusammenfassung" abbildet
    4. Ergebnis in Liste speichern

Der weitere Programmablauf ist dann wie folgt:

1. Eingabe überprüfen und in einzelne Elemente aufteilen
2. Operator identifizieren und anhand dieses, Wörter in zwei Listen (positiv und negativ) aufteilen
3. Für Elemente in positiver Liste Embeddings addieren, für negative subtrahieren
4. Das Ergebnis mit den zuvor berechneten Embeddings abgleichen (analog zu GloVe wieder ohne die Wörter, die in der Rechnung enthalten sind) und die fünf Wörter mit den ähnlichsten Embeddings ausgeben

#### Ergebnisse der Beispielrechnungen

| Rechnung                 | Ergebnis GloVe                                 | Ergebnis BERT                            |
| ------------------------ | ---------------------------------------------- | ---------------------------------------- |
| king - man + woman       | queen (Score: 0.8524)                          | lady (Score: 0.9683)                     |
| winter - cold + hot      | summer (Score: 0.8098)                         | spring (Score: 0.9501), summer Platz 2   |
| pizza - italy + japan    | sushi (Score: 0.6841)                          | trip (Score: 0.9551)                     |
| paris - france + germany | berlin (Score: 0.9204)                         | camp (Score: 0.9444)                     |
| paris - france + italy   | rome (Score: 0.8466)                           | afraid (Score: 0.9436), city auf Platz 2 |
| teacher - school + court | judge (Score: 0.8214), lawyer (Score: 0.8119)  | student (Score: 0.9549)                  |
| museum - art + science   | institute (Score: 0.8447), labroratory Platz 3 | garden (Score: 0.9005)                   |
| uncle - man + woman      | daughter (Score: 0.9003), aunt Platz 4         | lady (Score: 0.9644)                     |
| apple - iphone + galaxy  | milky (Score: 0.6767)                          | blue (Score: 0.9628)                     |

![](/words_calculator/charts/comparision.png)

Zu erkennen ist, dass der traditionelle Ansatz (GloVe) deutlich bessere Ergebnisse erzielt und lediglich bei einer besonders kontextabhängigen Rechnung, kein passendes Ergebnis in den fünf größten Übereinstimmungen zu finden ist. Bei dem Ansatz, in welchem die Transformer-Technologie verwendet wird, gibt es nur eine Rechnung, für die ein zufriedenstellendes Ergebnis erzielt wurde.

Diese Ergebnisse sind damit zu erklären, dass GloVe speziell für diese Art von Aufgaben trainiert und entwickelt wurde. Es werden kontextunabhängig schnell und zuverlässig die immer gleichen Vektoren für Wörter zurückgegeben, mit welchen direkt gerechnet werden kann. Für andere Anwendungsfälle ist das schlecht (Sitz(bank) und Geld(bank)), dafür jedoch sehr performant.

Transformer-Modelle sind, im Gegensatz dazu, speziell dafür entwickelt worden kontextsensitiv zu funktionieren. Für einfache Rechnung, wie sie in diesem Fall vorliegen, sind solche Modelle hingegen unpräzise, weshalb inkonsistente, bzw. ungenauere Ergebnisse erzielt wurden. Des weiteren, ist die Rechenzeit deutlich länger, um Embeddings zu erhalten, weshalb für diese Implementierung eine Optimierung vorgenommen wurde.

Insgesamt sind traditionelle Technologien wie GloVe für diese Art der Berechnungen deutlich besser geeignet, während Transformer-Modelle besser funktionieren, um Embeddings im Zusammenhang mit ganzen Sätzen oder Texten, zu berechnen.

### c)

Sinnvolle Rechenoperationen im Rahmen eines solchen Rechners sind die Addition und die Subtraktion, wie sie im Rahmen dieser Aufgabe bereits ausgeführt wurden.

Eine weitere Operation ist die Multiplikation, jedoch mit Zahlen anstelle von Wörtern. Das könnte zur Steigerung, beziehungsweise Minderung führen (Bsp.: 2 \* stark = stärker)

Andere Operationen, wie Multiplikation, Division, Poten-/Wurzelbildung sind nicht sinnvoll, da sie keine sinnvolle Interpretation/Bedeutung zulassen.
