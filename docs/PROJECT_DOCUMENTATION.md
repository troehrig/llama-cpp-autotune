# Projektdokumentation: Llama Autotune

## 1. Ziel und Motivation

Llama Autotune soll für ein lokales GGUF-Sprachmodell und eine vorhandene
llama.cpp-Installation automatisch sinnvolle Startparameter ermitteln. Das
Programm ist kein theoretischer Hardware-Rechner und verspricht kein globales
Optimum. Es soll mit vertretbarem Zeit- und Energieaufwand eine belastbare,
reproduzierbare Empfehlung erzeugen, die deutlich besser begründet ist als eine
manuell übernommene Parameterliste aus einem Forum.

Der wichtigste Zielbetrieb ist ein einzelner lokaler Chat beziehungsweise
Hermes Agent. In diesem Betrieb wächst der Gesprächskontext über längere Zeit.
Hermes komprimiert ihn zwischenzeitlich, sodass ein typischer Lebenszyklus aus
kleinen, mittleren und großen Kontextphasen besteht und anschließend wieder bei
einem kleineren Kontext beginnt.

Das System soll:

- Hardware, Modell und tatsächlich unterstützte llama.cpp-Optionen erkennen,
- nur auf der erkannten Umgebung sinnvolle Kandidaten bilden,
- Prozesse ohne Benutzereingriff starten, überwachen und sicher beenden,
- Fehler einzelner Kandidaten protokollieren und die Suche fortsetzen,
- Messwerte und Rohdaten dauerhaft nachvollziehbar ablegen,
- verschiedene Optimierungsziele transparent unterscheiden,
- eine deterministische Empfehlung mit direkt nutzbarem Startbefehl erzeugen,
- optional dasselbe lokale Modell zur verständlichen Erklärung einsetzen.

## 2. Abgrenzung

Llama Autotune ist kein universeller wissenschaftlicher Benchmark und kein
Ersatz für eine anwendungsspezifische Langzeitmessung. Insbesondere werden
derzeit nicht direkt gemessen:

- Qualität und Laufzeit der eigentlichen Hermes-Kontextkomprimierung,
- mehrere gleichzeitige Benutzer oder parallele Requests,
- Multi-GPU-Aufteilungen,
- Energieverbrauch über externe Leistungsmessgeräte,
- Antwortqualität verschiedener Quantisierungen oder Modelle,
- Betriebssystem- und Treibervergleiche über mehrere Rechner.

Ein Ergebnis gilt immer nur für die erkannte Kombination aus Hardware,
llama.cpp-Build, Modell und Testprofil.

## 3. Entstehung der Idee

Ausgangspunkt war ein Growing-Chat-Benchmark für Unsloth Desktop. Dieser konnte
einen laufenden `llama-server` erkennen, einen wachsenden Dialog erzeugen und
Promptverarbeitung, Generierung, Cachewiederverwendung und
Speculative-Decoding-Zähler messen.

Die zentrale neue Überlegung war, nicht nur eine bereits gestartete Instanz zu
vermessen. Stattdessen sollte das Programm llama.cpp selbst mit unterschiedlichen
Parametern starten, jeden Kandidaten isoliert prüfen, den Prozess wieder beenden
und aus den Ergebnissen eine Empfehlung ableiten. Dadurch wurde aus einem
Benchmark ein adaptives Optimierungssystem.

Der ursprüngliche Unsloth-Benchmark blieb während der Entwicklung als
Legacy-Komponente erhalten. Für die eigenständige öffentliche Veröffentlichung
wird er nicht übernommen: Der öffentliche Projektumfang bleibt dadurch klar,
und fremde Nutzer erhalten nur das allgemeine llama.cpp-Werkzeug.

## 4. Wesentliche Entwicklungsschritte und Überlegungen

### 4.1 Reproduzierbare Bestandsaufnahme

Die erste Stufe erfasst Betriebssystem, CPU-Threadzahl, RAM, NVIDIA-GPU,
Treiber, llama.cpp-Version und angebotene Kommandozeilenoptionen. Das Programm
verlässt sich nicht ausschließlich auf eine fest eingebaute Parameterliste,
sondern wertet die Hilfe der lokal vorhandenen Binärdateien aus.

Diese Entscheidung ist wichtig, weil sich llama.cpp schnell entwickelt und
Optionen hinzukommen, umbenannt werden oder in einem Build fehlen können.

### 4.2 Lokaler GGUF-Metadatenleser

Das heruntergeladene Hugging-Face-Modell lag als symbolischer Link auf eine
Blob-Datei ohne `.gguf`-Dateiendung vor. Eine frühe Pfadauflösung führte deshalb
zu einer falschen Ablehnung des gültigen Modells. Seitdem bleiben angegebener
und aufgelöster Pfad getrennt dokumentiert.

Für die Modellanalyse wurde ein kleiner GGUF-Metadatenleser ohne zusätzliche
Python-Pakete entwickelt. Er liest nur die benötigten skalaren Metadaten und
ausgewählte Tensornamen. Damit erkennt das Programm unter anderem:

- Modellname und Architektur,
- natives Kontextlimit,
- Layer-, Attention- und Embeddingdimensionen,
- Quantisierungsinformationen,
- MTP-/`nextn`-Tensoren,
- Hinweise auf Vision- oder weitere Spezialkomponenten.

### 4.3 Speicherplanung für wachsende Kontexte

Aus Modellmetadaten und freiem VRAM wird der ungefähre KV-Cache-Bedarf für
`f16`, `q8_0` und `q4_0` berechnet. Eine Sicherheitsmarge verhindert, dass der
gesamte freie Speicher verplant wird.

Die Berechnung ist eine Vorselektion und kein Ersatz für den realen Servertest.
Tatsächliche Backend-Allokationen, temporäre Puffer und andere GPU-Prozesse
können vom Schätzwert abweichen.

### 4.4 Smoke-Tests vor teuren Messungen

Ein kurzer `llama-bench`-Lauf prüft zuerst, ob Modell und Backend grundsätzlich
funktionieren. Erst danach folgen teurere Kandidaten. Das reduziert verlorene
Zeit bei falschen Pfaden, inkompatiblen Modellen oder fehlerhaften Builds.

Ein zusätzlicher Server-Smoke-Test prüft die OpenAI-kompatible Chat-API,
Reasoning- und Finalinhalt sowie die Wiederverwendung des Prompt-Caches.

### 4.5 Adaptive Suche statt vollständigem kartesischem Produkt

Ein vollständiges Produkt aller Batch-, Thread-, Cache-, Kontext- und
Spekulationsparameter würde schnell Tausende teure Langkontextläufe erzeugen.
Deshalb arbeitet das System stufenweise:

1. Smoke-Test,
2. Batch-/UBatch-/Flash-Attention-Screening,
3. Thread-Screening der erfolgreichen Kandidaten,
4. Kontext-/KV-Cache-Screening,
5. Spekulations-Screening,
6. wiederholte Finalvalidierung gegen passende Baselines.

Nur erfolgreiche Topkandidaten erreichen die jeweils teurere nächste Stufe.
Innerhalb einer Stufe werden die als sinnvoll erkannten Kombinationen trotzdem
vollständig geprüft.

### 4.6 Isolierte Serverprozesse und sichere Bereinigung

Jeder Serverkandidat läuft in einer eigenen Prozessgruppe mit lokal gebundenem
Port und temporärem API-Schlüssel. Das Programm wartet auf Bereitschaft, führt
die vorgesehenen Requests aus und beendet anschließend die gesamte
Prozessgruppe. Reagiert der Prozess nicht auf `SIGTERM`, folgt kontrolliert
`SIGKILL`.

Der API-Schlüssel wird in gespeicherten Kommandos redigiert. Serverlogs,
Requests, Responses und GPU-Snapshots bleiben zur Diagnose erhalten.

### 4.7 Adaptive Antwortbudgets und Formatfehler

Reasoning-Modelle können das anfängliche Antwortbudget vollständig für interne
Überlegungen verbrauchen, ohne finalen Inhalt zu liefern. Das ist kein
Performancefehler und darf nicht mit einer langsamen oder unpassenden
Parameterkombination verwechselt werden. Llama Autotune klassifiziert deshalb
HTTP-Fehler, abgeschnittenes Reasoning, abgeschnittenen finalen Inhalt und den
von llama.cpp bekannten `peg-gemma4`-Formatfehler getrennt.

Bei reinem, am Limit abgeschnittenem Reasoning oder einem PEG-Formatfehler wird
der Kandidat mit verdoppeltem Tokenbudget bis höchstens 2048 erneut ausgeführt.
Jeder Wiederholungsversuch startet einen vollständig neuen Serverprozess. Eine
Wiederholung in derselben Instanz würde den Prompt-Cache wiederverwenden und
damit Promptdurchsatz und Wall-Clock-Zeit verfälschen. Antwort, Serverlog,
effektives Tokenbudget und Retry-Grund bleiben je Versuch erhalten. Ist bereits
finaler Inhalt vorhanden, kann eine mit `length` beendete Antwort als nutzbare,
aber abgeschnittene Messung gewertet werden.

Ein nach Ausschöpfen der Retry-Stufen verbleibender PEG-Fehler wird nicht als
schlechter Parameterwert interpretiert, sondern als Runtime-Inkompatibilität
ausgewiesen. Die Fehlerklasse ist auch upstream in llama.cpp dokumentiert
([Issue #25072](https://github.com/ggml-org/llama.cpp/issues/25072),
[Issue #25986](https://github.com/ggml-org/llama.cpp/issues/25986)).

### 4.8 Wachsende Chats und Prompt-Cache

Der Dialog wird mit deterministischem technischem Fülltext auf definierte
Tokenziele erweitert. Vor jedem Request wird das Chat-Template angewendet und
über den Tokenize-Endpunkt die tatsächliche Modelltokenzahl bestimmt.

Erfasst werden unter anderem:

- gesamte Prompt-Tokens,
- wiederverwendete und neu ausgewertete Tokens,
- Prompt- und Generierungsdurchsatz,
- Wall-Clock-Zeit,
- Finish-Reason und finaler Inhalt,
- GPU-Zustand,
- Draft- und akzeptierte Spekulationstokens.

### 4.9 Spekulative Dekodierung und MTP

MTP-Kandidaten werden nur erzeugt, wenn sowohl das Modell passende Tensoren als
auch der lokale Server die erforderlichen Optionen meldet. Die Acceptance-Rate
ist ein Diagnosewert, aber kein eigenständiges Optimierungsziel. Entscheidend
sind tatsächlich gemessene Geschwindigkeit und Laufzeit.

Der Qwen3.8-Test zeigte, warum diese Trennung nötig ist: Eine Variante mit drei
Draft-Tokens beschleunigte die Generierung deutlich, verschlechterte aber bei
großen Kontexten teilweise die gesamte Wall-Clock-Zeit.

### 4.10 Passende `none`-Baseline

Eine Spekulationsvariante darf nicht gegen irgendeinen früheren Lauf verglichen
werden. Die Finalvalidierung führt deshalb für jeden übernommenen Kandidaten
eine `none`-Kontrolle mit derselben Batch-, Thread-, Cache- und
Kontextkonfiguration aus.

Dadurch wurde sichtbar:

- kurze Kontexte konnten mit MTP auch Ende-zu-Ende schneller sein,
- bei langen Kontexten blieb die Generierung schneller,
- die gesamte Antwortzeit konnte dort trotzdem schlechter werden.

### 4.11 Mehrere Optimierungsziele

Eine universell beste Konfiguration existiert nicht. Deshalb sind Suchumfang
und Bewertungsziel getrennt.

- `quick`, `balanced`, `thorough` bestimmen Kandidatenzahl und Wiederholungen.
- `hermes`, `balanced`, `interactive`, `long-context`, `throughput` bestimmen
  die Gewichtung derselben Messwerte.

Das Ziel `hermes` priorisiert Generierung und Antwortzeit, berücksichtigt aber
auch Promptarbeit, Stabilität und Speicherreserve. Kleine, mittlere und große
Kontexte werden gleich gewichtet. Das bildet näherungsweise einen wachsenden
Chat ab, der nach einer Komprimierung wieder kleiner beginnt. Die
Komprimierungsoperation selbst ist noch kein Benchmarkbestandteil.

### 4.12 Deterministische Empfehlung vor KI-Erklärung

Ranking, harte Erfolgskriterien und Startbefehl entstehen ausschließlich aus
Programmcode und Messwerten. Die lokale KI darf diese Entscheidung nicht
verändern.

Optional erhält das lokale Modell anschließend einen kompakten Datensatz und
formuliert eine verständliche Erklärung. Nach einem realen Test wurden dafür
zusätzliche Schutzregeln eingeführt:

- eine Wiederholung ist kein Stabilitätsnachweis,
- `parallel=1` erlaubt keine Aussage über Parallelbetrieb,
- Speicherwerte dürfen ohne passende Baseline nicht verglichen werden,
- Korrelationen dürfen nicht als technische Ursache ausgegeben werden,
- abgeschnittene Antworten werden automatisch kürzer wiederholt.

## 5. Bewertungslogik

Jede Finalkonfiguration erhält je Kontext einen normalisierten Score aus:

- Promptdurchsatz,
- Generierungsdurchsatz,
- inverser Wall-Clock-Zeit,
- Wiederholungsstabilität,
- freiem GPU-Speicher.

Das gewichtete geometrische Mittel über die Kontexte verhindert, dass ein sehr
guter Einzelwert einen schwachen Kontext vollständig verdeckt. Zusätzlich wird
der schlechteste Kontextscore ausgewiesen.

Seit Version 0.18 ist die geplante Kontextmenge die feste Bezugsgröße für die
Abdeckung. Schlägt beispielsweise die geplante 65k-Stufe fehl, wird sie nicht
aus dem Nenner entfernt. Die Empfehlung erhält dann den Status
`coverage-limited`, der Ausführungslauf endet als
`final-validation-partial`, der Score wird um die fehlende Abdeckung reduziert
und die Konfidenz bleibt unabhängig von der Wiederholungszahl `limited`.

Die Gewichte sind offen im Versuchsplan, Manifest und Bericht dokumentiert.
Damit bleibt eine Empfehlung überprüfbar und kann ohne neue Messungen unter
einem anderen Ziel neu bewertet werden.

## 6. Vereinfachte Bedienung

### 6.1 Vollständiger Hermes-orientierter Lauf

```bash
python3 run_autotune.py \
  /pfad/zu/llama.cpp/build/bin \
  /pfad/zum/modell.gguf
```

Standardmäßig verwendet der Runner:

- Suchprofil `quick`,
- Optimierungsziel `hermes`,
- automatische lokale KI-Erklärung,
- robuste Server- und Request-Timeouts,
- Laufdaten unter `./runs`.

Die letzte Ausgabezeile ist ein POSIX-Shell-sicherer, direkt kopierbarer
`llama-server`-Befehl. Pfade mit Leerzeichen werden korrekt gequotet. Für das
Ziel `hermes` setzt der Befehl – soweit vom nativen Modelllimit unterstützt –
`--ctx-size 131072`. Standardmäßig bindet er sicher an `--host 127.0.0.1`.
Für einen Hermes-Server im lokalen Netzwerk kann der Autotune-Lauf ausdrücklich
mit `--deployment-host 0.0.0.0` gestartet werden. Diese Freigabe über alle
Schnittstellen ist eine bewusste Betreiberentscheidung; Firewall-Regeln und bei
Bedarf eine API-Authentifizierung bleiben erforderlich.

Der Bericht unterscheidet dabei zwischen dem gewünschten Hermes-Betriebskontext
und der größten erfolgreich validierten Kontextstufe. Liegt für 131.072 Tokens
keine erfolgreiche Messung vor, bleibt der gewünschte Startwert erhalten, wird
aber ausdrücklich als nicht validiert gekennzeichnet.

Für eine belastbarere finale Prüfung:

```bash
python3 run_autotune.py \
  /pfad/zu/llama.cpp/build/bin \
  /pfad/zum/modell.gguf \
  --profile balanced
```

### 6.2 Vorhandenen Lauf zusammenfassen

Ohne Argument wird der neueste vollständige Lauf unter `./runs` gewählt:

```bash
python3 analyze_autotune.py
```

Ein bestimmter Lauf kann explizit angegeben werden:

```bash
python3 analyze_autotune.py runs/autotune_YYYYMMDD_HHMMSS
```

Die Ausgabe enthält Umgebung, Modell, Stufen, Kontextprofile,
Baselinevergleiche, alternative Zielgewinner, Status der KI-Erklärung,
wichtige Artefakte und als letzte Zeile erneut den Startbefehl.

## 7. Ergebnisstruktur

Jeder Lauf erhält einen eigenen Zeitstempelordner. Wichtige Dateien sind:

- `manifest.json`: vollständige Bestandsaufnahme und Fähigkeiten,
- `experiment_plan.json` und `.md`: adaptiver Versuchsplan,
- `autotune_state.json`: atomarer Zwischen- und Endzustand,
- `recommendation.json`: maschinenlesbare Empfehlung,
- `autotune_report.md`: deterministischer Gesamtbericht,
- `local_ai_analysis/`: Input, Requests, Responses, Bericht und Serverlog,
- Unterordner der einzelnen Screening- und Finalkandidaten.

Der Ordner `runs/` wird von Git ignoriert, weil er lokale Modellpfade, große
Messdaten und maschinenspezifische Informationen enthält.

## 8. Qualitätssicherung

Die Tests verwenden künstliche GGUF-Strukturen, simulierte Benchmarkantworten
und gemockte Serverprozesse. Geprüft werden unter anderem:

- GGUF-Parsing und ungültige Dateien,
- symbolische Modellpfade,
- Hardware- und Cacheplanung,
- adaptive Kandidatenauswahl,
- Prozessfehler und Timeouts,
- Ranking und Zielwechsel,
- passende `none`-Kontrollen,
- lokaler KI-Request, Retry und Prozessbereinigung,
- adaptive Tokenbudgets, Reasoning-Abbruch und Gemma-4-PEG-Fehler,
- Wiederverwendung vorhandener Messungen,
- Erstellung kopierbarer Shellbefehle,
- Zusammenfassung eines Laufordners.

Entscheidend bleiben ergänzende Realtests auf der Zielhardware. Der erste
vollständige Referenzlauf erfolgte mit einer RTX 4090 und einem lokalen
Qwen3.8-27B-GGUF-Modell.

## 9. Zusammenarbeit und Git-Vorgehen

Das Projekt wurde schrittweise auf einem Feature-Branch entwickelt. Jede
funktionale Erweiterung wurde lokal beziehungsweise in einer isolierten
Arbeitsumgebung geprüft, in kleine nachvollziehbare Commits aufgeteilt und vom
Benutzer auf der realen Hardware ausgeführt. Beobachtungen aus diesen Läufen
flossen in die jeweils nächste Version ein.

Konzeption, Prioritäten, Zielbetrieb, reale Modelltests und die Bewertung der
Ergebnisse entstanden in einer iterativen Zusammenarbeit zwischen menschlicher
Projektleitung und ChatGPT/Codex von OpenAI. Die KI unterstützte insbesondere
bei Architektur, Implementierung, Tests, Diagnose und Dokumentation. Die reale
Ausführung, Systemadministration und Ergebnisprüfung erfolgten durch einen
menschlichen Maintainer. KI-generierter Code und KI-Erklärungen wurden nicht als
ungeprüfte Autorität behandelt, sondern durch Tests, deterministische Regeln und
reale Messungen kontrolliert.

## 10. Bekannte Grenzen und nächste Schritte

Für weitere Modelle sind insbesondere folgende Punkte zu beobachten:

- Architekturmetadaten und KV-Cache-Schätzung,
- Modelle ohne MTP-Tensoren,
- andere Quantisierungen und Modellgrößen,
- VRAM-Grenzfälle und CPU-Offload,
- Chat-Templates mit abweichendem Reasoningverhalten,
- verbleibende Fehler des `peg-gemma4`-Parsers in llama.cpp,
- Unterschiede zwischen aktuellen llama.cpp-Versionen.

Spätere Erweiterungen können einen expliziten Hermes-Kompressionszyklus,
Resume-Funktion für abgebrochene Läufe, Energieverbrauchsmessung und getestete
Parallelitätsprofile ergänzen.
