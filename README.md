# Sonelgaz — système de prévision de consommation (prototype)

## Ce qui est sur GitHub (léger) — et ce qui ne l’est pas

Le dépôt contient **le code**, la **documentation**, `requirements.txt`, le script **`run_all.ps1`**, et **`models/README.md`**.

Il **ne contient pas** (et ne doit pas contenir) : le dossier **`.venv`**, le fichier **`sonelgaz_consumption_data.csv`**, ni les **gros fichiers entraînés** dans `models/` (`.pkl`, `.h5`, `.keras`, `*_architecture.json`). Ils sont listés dans **`.gitignore`**. Après un `git clone`, lancez **`run_all.ps1`** (voir ci-dessous) pour tout régénérer.

**Avant un `git push` :** fermez l’application (`app.py`) et toute fenêtre Python qui utilise ce dossier, sinon Windows peut **refuser de supprimer** `.venv` (fichiers verrouillés). Même sans supprimer `.venv` à la main, Git **l’ignore** s’il n’a jamais été commité.

---

## Démarrage ultra-simple (sans connaître l’IA)

Vous devez seulement avoir **Python installé** sur Windows (version **3.10 ou plus récente**).  
Téléchargement : [python.org/downloads](https://www.python.org/downloads/) — pendant l’installation, cochez **« Add python.exe to PATH »**.

### Option A — PowerShell (recommandé pour débuter)

1. Ouvrez le dossier du projet dans l’Explorateur Windows.
2. Dans la barre d’adresse, tapez **`powershell`** puis Entrée (une fenêtre PowerShell s’ouvre dans ce dossier).
3. Déverrouillez le script (souvent nécessaire après un clone, un zip ou un dossier OneDrive) :

```powershell
Unblock-File -Path .\run_all.ps1
```

4. Lancez le tout-en-un :

```powershell
.\run_all.ps1
```

Le script va, dans l’ordre :

- créer un dossier **`.venv`** (environnement Python isolé, une seule fois),
- installer les bibliothèques nécessaires,
- générer les données,
- entraîner les modèles (cela peut prendre **plusieurs minutes**),
- ouvrir l’**application graphique**.

### Option B — si PowerShell refuse le script (politique d’exécution / signature)

Pour votre utilisateur uniquement (une fois) :

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Ou lancement ponctuel sans changer la politique :

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

### Option C — 5 commandes à la main (si vous préférez tout taper)

Ouvrez **PowerShell** ou **Invite de commandes** **dans le dossier du projet**, puis copiez-collez **l’une après l’autre** :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe generate_data.py
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe app.py
```

*(La dernière ligne ouvre l’application. Les lignes d’avant ne sont à refaire que si vous changez le code ou les données.)*

---

## Résumé rapide

Ce dépôt implémente une **chaîne complète** hors ligne pour simuler la prévision de **consommation électrique** (kWh) pour un contexte type Sonelgaz :

1. **`generate_data.py`** — génère **2 ans** de données **synthétiques** horaires → `sonelgaz_consumption_data.csv`.
2. **`train_models.py`** — entraîne **4 modèles** (Random Forest, XGBoost, LSTM, Transformer), évalue (MAE / RMSE / R²) et enregistre les artefacts dans **`models/`**.
3. **`app.py`** — application **bureau** (CustomTkinter + Matplotlib) qui charge CSV + modèles pour le **tableau de bord**, les **prédictions** et un **aperçu télémétrie**.

**Logo** : placez `sonalgaz.webp` à la racine (déjà utilisé par l’UI).

**Important** : vous pouvez maintenant **assigner manuellement un modèle à chaque horizon** (Daily/Weekly/Monthly/Quarterly) dans `Forecast Center`. La sortie numérique principale reste une **prévision « prochaine heure »** (même cible qu’à l’entraînement). La courbe longue dans l’interface sert surtout à la **visualisation**.

---

## Arborescence utile

| Fichier / dossier | Rôle |
|-------------------|------|
| `generate_data.py` | Génération CSV synthétique |
| `train_models.py` | Entraînement + sauvegarde `models/` |
| `app.py` | Interface graphique |
| `sonelgaz_consumption_data.csv` | Données horaires (entrée training + app) |
| `models/` | Modèles et scalers (`models/README.md`) |
| `requirements.txt` | Dépendances pip |
| `run_all.ps1` | Démarrage tout-en-un (PowerShell ; voir `Unblock-File` dans la section démarrage) |
| `SYSTEM_DESIGN.md` | Conception (EN) + rapport de réalisation (FR) |

---

## Prérequis (rappel court)

- **Python 3.10+** sur Windows.
- Une connexion Internet la **première fois** uniquement (téléchargement des bibliothèques).

---

## Exécution manuelle (ordre des étapes)

À n’utiliser que si vous ne passez pas par `run_all.ps1` :

```text
generate_data.py   →  crée / met à jour le fichier CSV des données
train_models.py      →  crée les fichiers dans le dossier models/
app.py               →  ouvre l’interface utilisateur
```

---

## Interface (`app.py`)

- **Grid Dashboard** : indicateurs agrégés + courbe sur les 100 dernières heures.
- **Forecast Center** : choix du scénario d’UI + lancement inférence.
- **Telemetry Data** : derniers relevés en lecture seule.

---

## Documents

- `SYSTEM_DESIGN.md` — conception détaillée (EN) et rapport de réalisation (FR) dans un seul document.

## Auteur (projet original)

Chaghi Balsem
