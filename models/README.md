# Dossier `models/` — artefacts d’entraînement

Ce répertoire est **produit par** `train_models.py` et **consommé par** `app.py`.

## Fichiers attendus après un entraînement réussi

| Fichier | Description |
|--------|-------------|
| `rf_model.pkl` | RandomForestRegressor (scikit-learn), sérialisé avec `pickle`. |
| `xgb_model.pkl` | XGBoost regressor, sérialisé avec `pickle`. |
| `scaler_X.pkl` | `MinMaxScaler` ajusté sur les features d’entraînement. |
| `scaler_y.pkl` | `MinMaxScaler` sur la cible `Consumption`. |
| `lstm_architecture.json` | Graphe Keras LSTM exporté en JSON. |
| `lstm_weights.weights.h5` | Poids TensorFlow du LSTM (compatible `load_weights`). |
| `transformer_architecture.json` | Graphe Keras Transformer exporté en JSON. |
| `transformer_weights.weights.h5` | Poids TensorFlow du Transformer. |

## Fallback (anciennes versions)

Si seuls `lstm_model.h5` ou `transformer_model.h5` existent, `app.py` peut tenter de les charger comme modèles Keras complets (moins portable entre versions TF).

## À ne pas committer (optionnel)

Selon la politique du dépôt, vous pouvez exclure les gros binaires `.pkl` / `.h5` du contrôle de version et régénérer avec `python train_models.py`.
