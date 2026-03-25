import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

from features import compute_features, get_data


def walk_forward_validation(feat: pd.DataFrame,
                             n_train: int = 504,
                             n_test:  int = 126):
    """
    Entraîne et évalue XGBoost avec une validation walk-forward.
    feat    : DataFrame retourné par compute_features()
    n_train : nombre de jours d'entraînement (~2 ans)
    n_test  : nombre de jours de test (~6 mois)
    """
    feature_cols = [c for c in feat.columns if c != 'target']

    results = []
    models  = []

    for start in range(0, len(feat) - n_train - n_test, n_test):

        train = feat.iloc[start : start + n_train]
        test  = feat.iloc[start + n_train : start + n_train + n_test]

        X_train = train[feature_cols]
        y_train = train['target']
        X_test  = test[feature_cols]
        y_test  = test['target']

        model = xgb.XGBClassifier(
            n_estimators     = 100,
            max_depth        = 3,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            random_state     = 42,
            eval_metric      = 'logloss',
            verbosity        = 0
        )

        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        results.append({
            'debut'    : test.index[0].strftime('%Y-%m-%d'),
            'fin'      : test.index[-1].strftime('%Y-%m-%d'),
            'accuracy' : round(accuracy_score(y_test, y_pred), 3),
            'auc'      : round(roc_auc_score(y_test, y_pred_prob), 3),
            'n_train'  : len(train),
            'n_test'   : len(test)
        })

        models.append(model)

    results_df = pd.DataFrame(results)
    return results_df, models


def feature_importance(model, feature_cols: list) -> pd.Series:
    """
    Retourne les features classés par importance décroissante.
    """
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    return importance


if __name__ == '__main__':
    print("Chargement des données...")
    df   = get_data('SPY', '2018-01-01', '2024-01-01')
    feat = compute_features(df)

    print(f"Données : {len(feat)} jours, {len(feat.columns)} colonnes\n")

    print("Walk-forward validation en cours...")
    results, models = walk_forward_validation(feat)

    print("\nRésultats par période :")
    print(results.to_string(index=False))

    print(f"\nMoyennes :")
    print(f"  Accuracy moyenne : {results['accuracy'].mean():.3f}")
    print(f"  AUC moyenne      : {results['auc'].mean():.3f}")

    feature_cols = [c for c in feat.columns if c != 'target']
    print(f"\nFeature importance (dernier modèle) :")
    print(feature_importance(models[-1], feature_cols).to_string())