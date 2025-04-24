# Programmieraufgaben - Seminar zu Generativer KI 
Konrad Christoph Martens; Finnian

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

Auffällig ist, dass die durchschnittliche Anzahl der Tokens pro Satz mit zunehmender Vokabulargröße abnimmt. Außerdem erzielen die auf Deutsch bzw. Englisch trainierten Tokenizer die besten Ergebnisse für Sätze in der jeweiligen Sprache. Die Anzahl der Token für gemischte Sätze liegt für beide Tokenizer zwischen den Ergebnissen für deutsche und englische Sätze. Darüber hinaus kann der kombinierte Tokenizer bei einer Vokabulargröße von 500 gemischten Sätzen mit durchschnittlich 45,3 Tokens pro Satz besser kodieren, während bei höheren Vokabulargrößen die englischen Sätze bessere Ergebnisse erzielen. Insgesamt können englische Sätze über alle Vokabulargrößen hinweg mit durchschnittlich am wenigsten Tokens kodiert werden, während deutsche und gemischte Sätze vergleichbar viele Tokens benötigen.

### c)
Die Tokenanzahl pro Satz hängt direkt von der Vokabulargröße ab. Kleine Vokabulare (ca. 500 Einheiten) erzeugen mehr kurze Tokens, große Vokabulare (ca. 1500) weniger und längere Tokens, die ganze Wörter oder bedeutungsvolle Wortteile umfassen können. Dabei gilt: Kleine Vokabulare ermöglichen schnelleres Training bei ineffizientem Encoding, große Vokabulare verlangsamen das Training, bieten aber effizienteres Encoding. Die größten Effizienzgewinne liegen zwischen 500 und 1000 Vokabeleinheiten.

Die Effizienzunterschiede zwischen den Tokenizern spiegeln die sprachspezifischen Eigenschaften wider. Der deutsche Tokenizer hat beispielsweise gelernt, deutsche Komposita und morphologische Strukturen effizient zu kodieren, während der englische Tokenizer die häufigen englischen Wortbausteine besser repräsentiert. Beide zeigen deutliche Schwächen bei fremdsprachigen Texten.
Gemischte Sätze liegen in ihrer Tokenanzahl zwischen den Werten der spezialisierten Modelle. Interessanterweise erweist sich bei sehr kleinen Vokabularen (etwa 500 Sub-Wörter) ein kombinierter, multilingualer Tokenizer als vorteilhaft für gemischte Sätze, da er beide Sprachmuster abdeckt, wohingegen ab mittleren bis großen Vokabularen englische Sätze weniger Token mit dem kombinierten Tokenizer erfordern und somit am effizientesten kodiert werden.

Die Ergebnisse zeigen auch, dass eine größere Vokabulargröße zu einer kompakteren Darstellung von Informationen führt, da mehr bedeutungsvolle Subword-Einheiten in einem einzelnen Token erfasst werden können.

Der kombinierte deutsch-englische Tokenizer zeigt ausgewogene Leistung in beiden Sprachen. Er erreicht zwar nicht die Spitzeneffizienz der spezialisierten Modelle, benötigt aber weniger Tokens für fremdsprachliche Sätze als die jeweils unpassenden Einzelsprach-Tokenizer.

Die optimale Wahl hängt vom Anwendungsszenario ab:

- Einsprachige Systeme: Sprachspezifische Tokenizer mit großem Vokabular
- Mehrsprachige/gemischte Inhalte: Kombinierte Tokenizer für konsistente Ergebnisse über Sprachgrenzen hinweg
- Ressourcenbeschränkte Szenarien: Multilingualer Ansatz für beste Balance zwischen Kompaktheit und Flexibilität