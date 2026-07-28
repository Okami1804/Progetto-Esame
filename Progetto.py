import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE

# Caricamento dati da csv
df = pd.read_csv('dataset_batteri.csv')

# Definizione di X, y e groups. Non utilizziamo LabelEncoder perché la variabile target è già binaria
features = ['attractant_conc_uM', 'repellent_conc_uM', 'signal_delta']
X = df[features]
y = df['is_tumble']
groups = df['bacterium_id'] # Usiamo i gruppi per evitare che lo stesso batterio finisca sia nel train che nel test

# Lo Split lo basiamo sui Gruppi (cioè batteri interi in modo che non ci siano leak di informazioni tra train e test). Usiamo GroupShuffleSplit per fare uno split casuale ma basato sui gruppi.
# Test size 20% quindi l'80% dei batteri andrà nel train, il 20% nel test. In realtà risulta che per l'MLPClassifier (vedi dopo) viene preso automaticamente il 10% del train set come validation set per l'early stopping. Quindi il train set effettivo sarà l'80% * 90% = 72% dei batteri totali, mentre il test set sarà il 20% dei batteri totali.
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx] # Per Cross-Validation

# Implementazione dell'algoritmo evolutivo per ottimizzare gli iper-parametri della MLP, in particolare il numero di neuroni per i due layer nascosti, partendo da 5 configurazioni pre-impostate.
from deap import base, creator, tools
import random

# Definiamo gkf
gkf = GroupKFold(n_splits=5)  

# Inizio algoritmo evolutivo
# Creazione dinamica delle classi di base per DEAP, vedi dopo (riga 82)
if not hasattr(creator, "FitnessMax"): 
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

def evaluate_mlp(individual):
    # Estraiamo i neuroni dall'individuo. 
    # Usiamo max(4, ...) per garantire che una mutazione non generi layer negativi o da 0 neuroni.
    n1 = max(4, int(individual[0]))
    n2 = max(4, int(individual[1]))
    
    pipe_test = make_pipeline_imb(  # Serve perche la pipeline classica di sklearn non supporta SMOTE
        StandardScaler(),
        SMOTE(random_state=42),
        MLPClassifier(hidden_layer_sizes=(n1, n2), max_iter=300, random_state=42, early_stopping=True)
    )
    
    scores_fold = []
    for train_idx_f, test_idx_f in gkf.split(X_train, y_train, groups_train):
        pipe_test.fit(X_train.iloc[train_idx_f], y_train.iloc[train_idx_f])
        y_pred = pipe_test.predict(X_train.iloc[test_idx_f])
        scores_fold.append(f1_score(y_train.iloc[test_idx_f], y_pred))
        
    # In DEAP la funzione di valutazione deve restituire una tupla
    return (np.mean(scores_fold),)

def evaluate_invalid_individuals(population, toolbox):
    # Ricalcola la fitness solo per gli individui generati da incrocio o mutazione, lasciando inalterati quelli sopravvissuti per elitismo
    invalid_individuals = [] 
    for ind in population:
        if ind.fitness.valid == False: # Se non è valido
            invalid_individuals.append(ind) # Lo aggiungiamo alla lista
    fitnesses = map(toolbox.evaluate, invalid_individuals)
    
    for individual, fitness in zip(invalid_individuals, fitnesses):
        individual.fitness.values = fitness

# Configurazione del Toolbox
toolbox = base.Toolbox()

# Registriamo una funzione che sceglie un numero di neuroni casuale per l'inizializzazione
toolbox.register("attr_neurons", random.choice, [4, 8, 16, 32, 64, 128])    # Usiamo questi numeri gia' impostati per evitare configurazioni inutili e ridurre carico computazionale non necessario a calcolare tutti i casi possibili, per esempio considerando tutti i numeri fra 4 e 128

# Registriamo il generatore dell'individuo e della popolazione
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_neurons, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Registriamo gli operatori genetici
toolbox.register("evaluate", evaluate_mlp)
toolbox.register("select", tools.selTournament, tournsize=2)    # Selezione a torneo, per regolare facilmente la pressione selettiva e dare una probabilità bassa anche agli individui meno performanti di riprodursi
toolbox.register("mate", tools.cxOnePoint) # OnePointCrossover, dato che abbiamo solo due geni

def custom_mutate(individual, indpb):    #Mutazione personalizzata di tipo uniforme tra i valori consentiti, in modo da valutare ogni configurazione con lo stesso peso
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice([4, 8, 16, 32, 64, 128])
    return (individual,)

toolbox.register("mutate", custom_mutate, indpb=0.5)


import argparse
# Diamo la possibilità di scegliere liberamente gli iperparametri per il GA
def parse_args():
    parser = argparse.ArgumentParser(
        description="Script per la ricerca ottima di parametri per la rete neurale"
    )
    parser.add_argument("--population_size", type=int, default=5, help="Dimensione della popolazione")
    parser.add_argument("--num_generations", type=int, default=3, help="Numero di generazioni")
    parser.add_argument("--cx_prob", type=float, default=0.5, help="Probabilità di incrocio")
    parser.add_argument("--mut_prob", type=float, default=0.2, help="Probabilità di mutazione")
    return parser.parse_args()

args = parse_args()

# Parametri dell'esperimento
POP_SIZE = args.population_size
NUM_GENERATIONS = args.num_generations
CX_PROB = args.cx_prob
MUT_PROB = args.mut_prob
ELITE_SIZE = 1

# Generiamo la popolazione iniziale
population = toolbox.population(n=POP_SIZE)
hall_of_fame = tools.HallOfFame(maxsize=1)

print("\nInizio Cross-Validation ed Evoluzione (DEAP)")
history_best_fitness = [] #Liste per grafico di convergenza dell'algoritmo evolutivo
history_avg_fitness = []
# Valutazione della popolazione iniziale
evaluate_invalid_individuals(population, toolbox)
hall_of_fame.update(population)

# Ciclo Evolutivo
for generation in range(1, NUM_GENERATIONS + 1):
    print(f"Generazione {generation} in corso...")
    
    # Selezione dei genitori (clonandoli prima di modificarli)
    offspring = toolbox.select(population, len(population) - ELITE_SIZE)
    offspring = list(map(toolbox.clone, offspring))
    
    # Incrocio applicato a coppie consecutive
    for child1, child2 in zip(offspring[0::2], offspring[1::2]):
        if random.random() < CX_PROB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            
    # Mutazione applicata in modo indipendente
    for mutant in offspring:
        if random.random() < MUT_PROB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
    # Elitismo: copiamo il miglior individuo invariato
    elites = tools.selBest(population, ELITE_SIZE)
    elites = list(map(toolbox.clone, elites))
    
    # Valutiamo solo la prole modificata
    evaluate_invalid_individuals(offspring, toolbox)
    
    # Sostituzione generazionale
    population[:] = offspring + elites
    hall_of_fame.update(population)
    
    # Stampa di progresso
    best_gen = tools.selBest(population, 1)[0]
    print(f"Miglior individuo attuale: {best_gen} con F1: {best_gen.fitness.values[0]:.3f}")

    ## Storia della fitness per grafico di convergenza
    best_gen = tools.selBest(population, 1)[0]
    best_fit = best_gen.fitness.values[0]
    all_fits = [     
	    ind.fitness.values[0] for ind in population if ind.fitness.valid
    ]
    avg_fit = np.mean(all_fits) if all_fits else 0.0

    history_best_fitness.append(best_fit)
    history_avg_fitness.append(avg_fit)

# Estrazione del campione finale
best_individual = hall_of_fame[0]
best_n1 = max(4, int(best_individual[0]))
best_n2 = max(4, int(best_individual[1]))
miglior_configurazione = (best_n1, best_n2)

print(f"\nL'evoluzione con DEAP ha scelto la struttura: {miglior_configurazione}\n")
# Fine algoritmo evolutivo

# Creazione della Pipeline, con libreria imbalanced-learn approfondita per provare a gestire il problema di classe estremamente sbilanciata
pipe_mlp = make_pipeline_imb(   # Creiamo una pipeline con questo metodo per poter utilizzare SMOTE, che non è disponibile nella pipeline classica di sklearn. La pipeline è composta da tre step: StandardScaler, SMOTE e MLPClassifier.
    StandardScaler(), # Applichiamo StandardScaler per normalizzare le feature, in modo da dare lo stesso peso a tutte le feature e migliorare la convergenza della rete neurale.
    SMOTE(random_state=42), # Applichiamo SMOTE per bilanciare le classi nel train set. SMOTE genera nuovi campioni della classe minoritaria, nel nostro caso lo stato is_tumble=1, per ridurre lo sbilanciamento.
    MLPClassifier(  # Setta una MLP con due hidden layer, max_iter=500 e early stopping
        hidden_layer_sizes=miglior_configurazione,  # Il numero di neuroni nello strato nascosto è quello ottimale trovato dall'algoritmo evolutivo.
        max_iter=500, 
        random_state=42,
        early_stopping=True # Termina l'addestramento se la validazione non migliora per 10 iterazioni consecutive. Secondo la descrizione utilizza automaticamente il 10% del train set come validation set per l'early stopping, quindi non l'abbiamo impostato manualmente.
    )
)

from sklearn.metrics import precision_recall_curve

# Cross-validation sul Train Set e generazione predizioni Out-Of-Fold (OOF) per evitare data leakage
print("\nInizio Cross-Validation (GroupKFold)")
# Usiamo GroupKFold a 5 split, già definito sopra, per fare una cross-validation basata sui gruppi (batteri) e calcolare l'F1-Score e le probabilità OOF. Questo ci permette di valutare la robustezza del modello e la sua capacità di generalizzare su batteri non visti durante l'addestramento.

oof_probas = np.zeros(len(X_train)) # Array per salvare le probabilità Out-Of-Fold per tutto il train set
scores = [] # Lista per salvare gli F1-Score di ogni fold

for k, (train, test) in enumerate(gkf.split(X_train, y_train, groups_train)):   # Iteriamo su tutti i fold
    # Addestramento sul fold
    pipe_mlp.fit(X_train.iloc[train], y_train.iloc[train])  # Addestriamo la pipeline sul fold corrente
    
    # Valutazione sul fold: otteniamo le probabilità predette per la classe positiva (is_tumble=1) sui dati di validazione del fold
    y_proba_fold = pipe_mlp.predict_proba(X_train.iloc[test])[:, 1]
    oof_probas[test] = y_proba_fold # Salviamo le probabilità out-of-fold
    
    # Convertiamo le probabilità in predizioni binarie usando la soglia standard 0.5 per il log del fold
    y_pred_fold = (y_proba_fold >= 0.5).astype(int)
    
    # Calcoliamo l'F1-Score per la classe 1
    score = f1_score(y_train.iloc[test], y_pred_fold)   # Calcoliamo l'F1-Score per la classe 1
    scores.append(score)
    # Stampa dei risultati per il fold corrente
    print(f'Fold: {k+1:02d}, '
          f'Tumble reali: {np.sum(y_train.iloc[test] == 1)}, '
          f'F1-Score Tumble (soglia 0.5): {score:.3f}')

# Stampa della media e deviazione standard degli F1-Score dei fold
print(f'\nF1-Score Tumble Medio (soglia 0.5): {np.mean(scores):.3f} +/- {np.std(scores):.3f}')

# Calcoliamo Precision, Recall e le Soglie corrispondenti sulle probabilità Out-Of-Fold (senza data leakage)
precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probas)

# Calcoliamo l'F1-Score per ogni soglia (evitando divisioni per zero)
f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls), out=np.zeros_like(precisions), where=(precisions + recalls) != 0)

# Troviamo la soglia che ha generato l'F1-Score più alto sulle predizioni Out-Of-Fold
indice_migliore = np.argmax(f1_scores)
soglia_ottimale = thresholds[indice_migliore]
f1_massimo = f1_scores[indice_migliore]

# Stampa dei risultati dell'ottimizzazione della soglia
print(f"\nOttimizzazione della Soglia (su predizioni Out-Of-Fold)")
print(f"Soglia Matematica Ottimale: {soglia_ottimale:.5f}")
print(f"F1-Score Massimo Raggiungibile (in CV): {f1_massimo:.3f}\n")

# Addestramento della pipeline sull'intero train set finale
pipe_mlp.fit(X_train, y_train)

# Impostiamo la soglia ottimale per la predizione sul test set
soglia = soglia_ottimale

# Utilizziamo questo invece di pipe_mlp.predict(X_test) per ottenere le probabilità predette per la classe positiva (cioè is_tumble=1)
y_proba_test = pipe_mlp.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= soglia).astype(int)  # Predizione finale sul test set usando la soglia ottimale

print("Risultati sul Test Set")
print(confusion_matrix(y_test, y_pred_test))    # Stampa la Matrice di confusione
print(classification_report(y_test, y_pred_test))  # Stampa il Report di classificazione, tutte le metriche principali di classificazione
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Definiamo il target continuo, ora la probabilità di tumble
y_reg = df['tumble_probability']
y_train_reg = y_reg.iloc[train_idx] # Creiamo il target continuo per il train set
y_test_reg = y_reg.iloc[test_idx]

# Creiamo la pipeline di base per gli input X
pipe_base = make_pipeline(  # Ora utilizziamo il comando per pipeline classico dato che non abbiamo bisogno di SMOTE
    StandardScaler(),   # StandardScaler come nel caso precedente per normalizzare le feature
    MLPRegressor(   # Impostiamo una MLPRegressor con due hidden layer, 32 e 16 neuroni, max_iter=1000 e tol=1e-6 (piu alto della default 1e-4 per avere una convergenza più precisa)
        hidden_layer_sizes=(32, 16),
        max_iter=1000,
        tol=1e-6,
        random_state=42
    )
)

'''Utilizziamo TransformedTargetRegressor con StandardScaler per normalizzare la variabile target (tumble_probability). 
Anche se lo StandardScaler non modifica la forma della distribuzione (non rimuove l'asimmetria),
portare il target a media 0 e varianza 1 stabilizza l'addestramento dell'MLPRegressor,
favorendo una convergenza più rapida e fluida dei gradienti.'''
# Questo scalerà Y in fase di fit() e farà l'inverse_transform in fase di predict()
model_regresso_ottimizzato = TransformedTargetRegressor(    
    regressor=pipe_base,
    transformer=StandardScaler()
)

# Addestramento
model_regresso_ottimizzato.fit(X_train, y_train_reg)

# Predizione sul test set
y_pred_reg = model_regresso_ottimizzato.predict(X_test)

# Valutazione
r2 = r2_score(y_test_reg, y_pred_reg)   # Calcoliamo l'R² Score, che indica quanto bene il modello spiega la variabilità del target continuo.
mse = mean_squared_error(y_test_reg, y_pred_reg)    # Calcoliamo l'errore quadratico medio MSE, che misura la media degli errori al quadrato tra le predizioni e i valori reali del target continuo.

print("Risultati Regressione (Con Target Scalato):")
print(f"R² : {r2:.6f}") # Come vediamo otteniamo un R² molto alto con MSE praticamente nullo, questo è dovuto ad una intrinseca relazione fra il signal_delta e probabilità di tumble, dato che il batterio quando si muove in direzione del gradiente di attrazione (o lontano dal gradiente di repulsione) ha una probabilità molto bassa di tumble, mentre quando si muove in direzione opposta al gradiente di attrazione (o verso il gradiente di repulsione) ha una probabilità molto alta di tumble.
print(f"MSE : {mse:.10f}")

'''
# ==========================================
# --- GRAFICI PER LA PRESENTAZIONE 
# ==========================================
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


fig_size = (8, 5)
# --- 1. Matrice di Confusione   ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_test,
    cmap='Blues',
    display_labels=['No Tumble', 'Tumble'],
    ax=ax,
)
ax.set_title(
    'Matrice di Confusione sul Test Set', fontsize=12, fontweight='bold'
)
ax.set_xlabel('Predizione', fontsize=10)
ax.set_ylabel('Valore Reale', fontsize=10)
ax.grid(False)  
plt.tight_layout()
plt.savefig('matrice_confusione.png', dpi=300)
plt.close()

# --- 2. Curva Precision-Recall e Soglia Ottimale ---

plt.figure(figsize=fig_size)
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1-Score', linewidth=2.5)
plt.axvline(
    x=soglia_ottimale,
    color='black',
    linestyle=':',
    label=f'Soglia Ottimale ({soglia_ottimale:.3f})',
)
plt.title(
    'Ottimizzazione della Soglia Decisionale', fontsize=12, fontweight='bold'
)
plt.xlabel('Soglia di Probabilità', fontsize=10)
plt.ylabel('Punteggio Metrica', fontsize=10)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('ottimizzazione_soglia.png', dpi=300)
plt.close()

# --- 3. Regressione: Valori Reali vs Predetti (R²) ---
plt.figure(figsize=fig_size)
plt.scatter(
    y_test_reg, y_pred_reg, alpha=0.5, color='#10b981', edgecolors='k', s=30
)
plt.plot(
    [y_test_reg.min(), y_test_reg.max()],
    [y_test_reg.min(), y_test_reg.max()],
    'r--',
    lw=2,
    label='Predizione Perfetta (y=x)',
)
plt.title(
    f'Regressione tumble_probability (R² = {r2:.4f})',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Probabilità Reale', fontsize=10)
plt.ylabel('Probabilità Predetta', fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig('regressione_r2.png', dpi=300)
plt.close()

# --- 4. Convergenza dell'Algoritmo Evolutivo ---
plt.figure(figsize=fig_size)
plt.plot(
    range(1, NUM_GENERATIONS + 1),
    history_best_fitness,
    'o-',
    color='#2563eb',
    label='Miglior Fitness (F1-Score)',
    linewidth=2,
)
plt.plot(
    range(1, NUM_GENERATIONS + 1),
    history_avg_fitness,
    's--',
    color='#94a3b8',
    label='Fitness Media',
    linewidth=1.5,
)
plt.title(
    "Convergenza dell'Algoritmo Evolutivo (DEAP)",
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Generazione', fontsize=10)
plt.ylabel('F1-Score', fontsize=10)
plt.xticks(range(1, NUM_GENERATIONS + 1))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('convergenza_ga.png', dpi=300)
plt.close()


print('Tutti i grafici per la presentazione sono stati salvati con successo!') '''
