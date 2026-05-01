# System design — Sonelgaz consumption prediction (prototype)

## 1. Purpose

Build an **end‑to‑end offline pipeline**: synthetic hourly data → train several regressors → run a **desktop UI** that loads artefacts and renders charts + a single‑step prediction. This document matches the **current repository** behaviour.

## 2. Logical architecture

Three cooperating pieces:

| Module | Responsibility |
|--------|----------------|
| **Data generation** (`generate_data.py`) | Emulate SCADA‑like series when real data is unavailable. |
| **AI training** (`train_models.py`) | Fit models + `MinMaxScaler` on features/target, persist to `models/`. |
| **Desktop app** (`app.py`) | Load CSV + pickles + Keras weights, display dashboard & run inference. |

## 3. Data layer

**File**: `sonelgaz_consumption_data.csv`

- **Granularity**: one row per hour (default **17 520** rows ≈ two years).
- **Target**: `Consumption` (kWh).
- **Features used in training / inference** (see `FEATURES` in code):  
  `Hour`, `DayOfWeek`, `Month`, `Season`, `Temperature`, `Current`, `IsHoliday`.
- **Extra column**: `Year` is generated for traceability but **not** part of `FEATURES` today.

**Pre‑processing in training**

- Chronological **80 % / 20 %** split (no shuffle).
- **MinMax scaling** for `X` and `y`; the same scaler objects are pickled for the app.

## 4. Model layer

All models are trained to predict **the next hour’s consumption** from the available inputs (either the last row’s feature vector or a **24‑hour sequence** for deep nets).

| Model | Library | Role in **UI** (label) | Notes |
|-------|---------|------------------------|-------|
| Random Forest | scikit‑learn | “Daily” | Tabular |
| XGBoost | xgboost | “Weekly” | Tabular |
| LSTM | TensorFlow/Keras | “Monthly” | Sequence length 24 |
| Transformer | TensorFlow/Keras | “Quarterly” | One encoder block + pooling |

**UI nuance**: horizon names in the app are **model selectors**, not separate dataset aggregations. A roadmap improvement would be true multi‑horizon targets or recursive forecasting.

**Metrics** (printed to console during `train_models.py`): MAE, RMSE, R² on the held‑out tail.

## 5. Persistence (`models/`)

See `models/README.md` for the authoritative file list.

**Design choice**: deep models are saved as **JSON architecture + `.weights.h5`** to reduce breakage across TensorFlow versions compared to monolithic `.h5` full models.

## 6. Desktop application

**Stack**: Python, **CustomTkinter**, **Matplotlib** (embedded via `FigureCanvasTkAgg`), **Pillow** for optional logo.

**Screens** (navigation unchanged across iterations):

1. Dashboard — KPIs + recent utilisation chart.  
2. Forecast — pick UI horizon, run model, show scalar prediction + illustrative long curve.  
3. Telemetry — tabular preview of recent measurements.

**Visualisation note**: long‑horizon plots in the app are **stylised helpers** (not strict autoregressive rollouts of the neural nets).

## 7. Technology summary

- **Language**: Python 3.x  
- **ML**: scikit‑learn, XGBoost, TensorFlow/Keras  
- **Data**: pandas, NumPy  
- **UI / plots**: CustomTkinter, Matplotlib, Pillow  

## 8. Future extensions (ideas)

- Real calendar holidays instead of random `IsHoliday` flags.
- Dedicated evaluation report (JSON/HTML) persisted next to checkpoints.
- True multi‑step forecasting UI aligned with horizons.
- Optional packaging (PyInstaller) for one‑click Windows distribution.

---

# Rapport 3 — Contribution partie réalisation *(FR)*

## 1. Introduction

Ce document décrit la **réalisation technique** du prototype de prévision de consommation électrique pour Sonelgaz : outils utilisés, **enchaînement des trois modules** (données → entraînement → application bureau), et extraits de code **alignés avec la version actuelle** du dépôt.

Pour une vue synthétique en anglais + commandes d’exécution : voir `README.md`.  
Pour le découpage fonctionnel détaillé en anglais : voir la **première partie** de ce fichier (sections *Purpose* à *Future extensions*).

## 2. Outils et logiciels de programmation

| Domaine | Technologies |
|---------|----------------|
| Langage | Python 3.x |
| Données | `pandas`, `numpy` |
| ML classique | `scikit-learn` (Random Forest), `xgboost` |
| Deep learning | `tensorflow` / Keras (LSTM, bloc Transformer léger) |
| Interface bureau | `tkinter`, `customtkinter` |
| Visualisation embarquée | `matplotlib` (`FigureCanvasTkAgg`) |
| Image logo | `Pillow` |
| Dépendances | `requirements.txt`, installation via `pip` |

*(L’ancienne mention Plotly dans une version antérieure du design a été abandonnée : les graphiques intégrés utilisent Matplotlib.)*

## 3. Réalisation de l’application prototype

Le prototype comporte **trois modules principaux** :

1. Génération de données synthétiques — `generate_data.py`  
2. Entraînement et persistance des modèles — `train_models.py`  
3. Application graphique — `app.py`

Les artefacts appris sont stockés sous `models/` (voir `models/README.md`).

### 3.1. Module de génération de données (`generate_data.py`)

**Rôle** : produire `sonelgaz_consumption_data.csv`, série **horaire sur deux ans** (17 520 lignes par défaut), avec variables temps, météorologie simulée, courant, consommation cible et jours fériés simplifiés.

**Principales étapes dans le code** :

- création de la plage temporelle `pandas` ;
- dérivation de `Hour`, `DayOfWeek`, `Month`, `Year`, `Season` ;
- signaux sinusoïdaux bruités pour température et courant ;
- agrégation d’effets (thermiques, intra‑journaliers, weekends, saison) pour `Consumption` ;
- tirage pseudo‑aléatoire contrôlé (`numpy.random.default_rng`) pour les fériés.

**Extrait minimal (structure)** :

```python
def generate_sonelgaz_data(start_date="2024-01-01", periods=17520, random_seed=42):
    rng = np.random.default_rng(random_seed)
    date_range = pd.date_range(start=start_date, periods=periods, freq="h")
    df = pd.DataFrame({"Timestamp": date_range})
    # … ingénierie des colonnes puis export CSV …
```

Le fichier CSV complet peut être régénéré à tout moment sans toucher au code ML.

### 3.2. Module d’entraînement (`train_models.py`)

**Rôle** :

- lire le CSV ;
- découper **chronologiquement** 80 % / 20 % ;
- **MinMaxScaler** sur `X` et `y` (mêmes objets utilisés plus tard dans `app.py`) ;
- entraîner RF, XGB, LSTM (fenêtre 24 h), Transformer ;
- afficher MAE, RMSE, R² sur la partie test ;
- sauvegarder `*.pkl` + graphes Keras en **JSON + poids `.weights.h5`** (plus robuste que d’anciens `.h5` monolithiques).

**Extrait minimal (préparation + sauvegarde)** :

```python
FEATURES = ["Hour", "DayOfWeek", "Month", "Season", "Temperature", "Current", "IsHoliday"]
TARGET = "Consumption"
# split_idx = int(len(df) * 0.8)  # ordre temporel respecté
# scaler_X.fit sur train seulement, idem scaler_y
# sauvegarde : rf_model.pkl, xgb_model.pkl, lstm_architecture.json + lstm_weights.weights.h5, etc.
```

### 3.3. Application de bureau (`app.py`)

**Rôle** : console « Grid Intelligence » — chargement des modèles, affichage du tableau de bord, lancement d’inférence, aperçu télémétrie. L’identité visuelle s’appuie sur les couleurs extraites du logo `sonalgaz.webp`.

**Navigation (inchangée fonctionnellement)** :

- **Grid Dashboard** — synthèse + courbe des 100 dernières heures.
- **Forecast Center** — menu d’horizon (*Daily / Weekly / Monthly / Quarterly*) qui **sélectionne le modèle** ; la valeur affichée reste une **prévision de charge pour l’heure suivante** (cohérent avec l’entraînement). La courbe longue est une **visualisation d’ambiance**, pas un déploiement récursif complet du réseau.
- **Telemetry Data** — extrait des dernières lignes du CSV.

**Extrait minimal (inférence tabulaire vs séquentielle)** :

```python
X_input = scaler_X.transform(data.tail(24)[FEATURES])
# RF / XGB : dernière ligne
pred = scaler_y.inverse_transform(model.predict(X_input[-1:]).reshape(-1, 1))[0, 0]
# LSTM / Transformer : tenseur (1, 24, n_features)
# pred = scaler_y.inverse_transform(model.predict(X_input.reshape(1, 24, -1), verbose=0))[0, 0]
```

## 4. Chaîne d’exécution recommandée

```text
python generate_data.py   →  sonelgaz_consumption_data.csv
python train_models.py    →  models/*
python app.py             →  interface utilisateur
```

## 5. Conclusion

Le prototype intègre génération synthétique, entraînement multi‑modèles et interface bureau cohérente avec les artefacts persistés. La documentation (`README.md`, ce fichier — conception en anglais et rapport de réalisation en français —, `models/README.md`) et les **docstrings** dans le code source complètent ce rapport pour faciliter la maintenance et les évolutions (prévision multi‑pas, calendrier de fériés réel, export d’évaluation, etc.).
