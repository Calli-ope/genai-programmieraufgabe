# Programmieraufgaben - Seminar zu Generativer KI 
Konrad Christoph Martens; Finnian

## 01 - BPE-Tokenizer
### b) + c)
Für den Vergleich wurden drei Tokenizer (Deutsch, Englisch, Deutsch + Englisch) auf dem verlinkten Datensatz des Auswärtigen Amtes trainiert. Dabei wurden drei Versionen mit unterschiedlicher Vokabulargröße (500, 1000, 1500) trainiert. 

Im Rahmen der Analyse wurden alle Versionen der Tokenizer mit jeweils drei deutschen, englischen und gemischten Sätzen getestet, die jeweils die gleiche Aussage enthalten und in der [Konfigurationsdatei](bpe_tokenizer/config.py) zu finden sind. Für die Charts wurde die durchschnittliche Anzahl der Tokens pro Satz berechnet, die auf der y-Achse gezeigt wird. Für jede Vokabulargröße werden alle drei Tokenizer für alle drei Sprachen verglichen.

**Vokabulargröße: 500**

![](bpe_tokenizer/charts/tokenizer_avg_comparison_vocab500.png)

**Vokabulargröße: 1000**

![](bpe_tokenizer/charts/tokenizer_avg_comparison_vocab1000.png)

**Vokabulargröße: 1500**

![](bpe_tokenizer/charts/tokenizer_avg_comparison_vocab1500.png)

**Erkenntnisse**

Die durschnittliche Anzahl der Tokens pro Satz hängt von der Vokabulargröße ab. Kleine Werte wie z.B. 500 führen zu mehr und kürzeren Tokens, die weniger Buchstaben umfassen, während höhere Werte wie z.B. 1500 weniger und dafür längere Tokens bedeuten, die ganze Worte oder Wortteile umfassen können. Der Trade-off zwischen einem kleinen Vokabular, was schnelleres Training aber dafür ineffizientreres Encoding bedeutet und einem großen Vokabular, was wiederum langsameres Training aber effizienteres Encoding mit sich bringt, ist auch in den Charts zu erkennen.

Es lässt sich erkennen, das die Tokenizer, die auf Deutsch und Englisch trainiert wurden, für Sätze in den jeweiligen Sprache über alle Vokabulargrößen hinweg besser abschneiden. Mit wachsender Vokabulargröße sinkt außerdem die Anzahl der benötigten Tokens.

Im Kontext des BPE-Algorithmus (Byte-Pair Encoding) sind diese Ergebnisse besonders aufschlussreich. Der BPE-Algorithmus funktioniert, indem er häufig vorkommende Zeichenpaare iterativ zusammenfügt, um ein Subword-Vokabular zu erstellen. Die Diagramme zeigen, dass die Tokenizer, die auf spezifischen Sprachen trainiert wurden, die statistischen Muster dieser Sprachen effektiv lernen konnten.
Die Effizienzunterschiede zwischen den Tokenizern spiegeln die sprachspezifischen Eigenschaften wider. Der deutsche Tokenizer hat beispielsweise gelernt, deutsche Komposita und morphologische Strukturen effizient zu kodieren, während der englische Tokenizer die häufigen englischen Wortbausteine besser repräsentiert. Die Ergebnisse zeigen auch, dass eine größere Vokabulargröße zu einer kompakteren Darstellung von Informationen führt, da mehr bedeutungsvolle Subword-Einheiten in einem einzelnen Token erfasst werden können.

### c)
1. Sprachspezifische Effizienz: Jeder Tokenizer zeigt die beste Leistung in seiner eigenen Sprache. Der englische Tokenizer benötigt beispielsweise durchgängig weniger Tokens für englischen Text im Vergleich zu deutschem Text und umgekehrt.

2. Einfluss der Vokabulargröße: Mit zunehmender Vokabulargröße von 500 auf 1500 nimmt die Anzahl der benötigten Tokens pro Satz bei allen Tokenizern und Sprachkombinationen ab. Dies zeigt, wie größere Vokabulare eine effizientere Kodierung ermöglichen, indem sie häufige Subword-Muster erfassen.

3. Sprachübergreifende Leistung: Der deutsche Tokenizer zeigt schwächere Leistung bei englischem Text und benötigt deutlich mehr Tokens als bei der Verarbeitung von deutschem Text. Ähnlich verhält es sich beim englischen Tokenizer für deutschen Text.

4. Vorteile des kombinierten Tokenizers: Der kombinierte Tokenizer zeigt eine ausgewogenere Leistung in beiden Sprachen, obwohl er nicht so effizient ist wie sprachspezifische Tokenizer in ihrer jeweiligen Sprache. Er ist jedoch besser als einzelsprachige Tokenizer bei nicht-nativen Sprachen.

5. Abnehmender Grenznutzen: Die Effizienzgewinne durch die Erhöhung der Vokabulargröße zeigen einen abnehmenden Grenznutzen. Die Reduzierung der Tokens pro Satz beim Übergang von 1000 auf 1500 Vokabulargröße ist weniger dramatisch als von 500 auf 1000.

Die effizienteste Informationsrepräsentation hängt vom Anwendungsfall ab:

- Für einsprachige Anwendungen bieten sprachspezifische Tokenizer mit größeren Vokabularen die effizienteste Kodierung

- Für mehrsprachige Anwendungen bietet ein kombinierter Tokenizer den besten Kompromiss, mit angemessener Effizienz über verschiedene Sprachen hinweg ohne extreme Ineffizienz in einer bestimmten Sprache