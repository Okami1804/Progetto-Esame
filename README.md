# Progetto-Esame

Repo contenente il progetto d'esame di Vincenzo Trani e Giuseppe Rossagno: Analisi Computazionale della Chemiotassi Batterica


Questo progetto analizza e modella il movimento chemiotattico dei batteri (*E. coli*) a partire da dati temporali di simulazione. L'obiettivo principale è studiare come il batterio risponde ai gradienti chimici ambientali, affrontando il problema con due approcci: la **classificazione dello stato istantaneo** e la **regressione della probabilità di movimento**.

---

## 📁 Struttura del Repository

* `dataset_batteri.csv`: Dataset con le traiettorie e le misurazioni fisiche dei batteri nel tempo.
* `Progetto.py`: Script principale Python contenente la pipeline completa.
## File dei grafici per la presentazione:
    * `convergenza_ga.png`: Convergenza dell'Algoritmo Evolutivo
    * `ottimizzazione_soglia.png`: Curva Precision-Recall e Soglia Ottimale
    * `matrice_confusione.png`: Matrice di Confusione
    * `regressione_r2.png`: Regressione: Valori Reali vs Predetti (R²)

---

## 🛠️ Requisiti

Installa le librerie necessarie con:

```bash
pip install pandas numpy scikit-learn imbalanced-learn deap matplotlib
```



## Personalizzazione

è possibile avviare lo script con i parametri predefiniti o personalizzare l'algoritmo genetico ad esempio con: 
```bash
python "Progetto.py" --population_size 10 --num_generations 10 --cx_prob 0.7 --mut_prob 0.2
```

La parte di codice che stampa i grafici utilizzati nella presentazione si trovano alla fine del progetto, virgolettati


## Considerazioni

Dal progetto emerge un comportamento molto diverso tra i due compiti di machine learning:
Predire lo stato momento per momento si è dimostrato un problema complesso. Questa difficoltà non deriva dall'algoritmo scelto o da un bilanciamento errato, che è stato gestito applicando SMOTE con la libreria imbalanced-learn, ma dalla componente stocastica del fenomeno biologico. L'E. coli infatti regola la probabilità di ruotare in base al gradiente chimico, ma il singolo evento di tumble mantiene un margine di imprevedibilità.
La stima della probabilità continua ha dato invece ottimi risultati, registrando un R^2 elevato e un errore MSE trascurabile. L'MLPRegressor è riuscito a catturare la dipendenza diretta tra la variazione del segnale (signal_delta) e la tendenza del batterio a cambiare direzione, confermando che il modello ha appreso correttamente la dinamica chemiotattica reale.