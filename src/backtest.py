import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import compute_features, get_data
from model import walk_forward_validation


def compute_pnl(feat: pd.DataFrame, models: list, n_train: int = 504, n_test: int = 126) -> pd.DataFrame:
    """
    Calcule le P&L journalier de la stratégie sur toutes les périodes de test.
    """
    feature_cols = [c for c in feat.columns if c != 'target']
    all_pnl = []

    for i, start in enumerate(range(0, len(feat) - n_train - n_test, n_test)):

        test = feat.iloc[start + n_train : start + n_train + n_test]

        X_test      = test[feature_cols]
        prob        = models[i].predict_proba(X_test)[:, 1]

        # Signal : +1 si prob > 0.5, -1 sinon
        signal      = np.where(prob > 0.5, 1, -1)

        # P&L : signal × return réel du lendemain
        # log_ret contient déjà le return de CE jour
        # on veut le return du jour SUIVANT dans le test set
        returns     = test['log_ret'].values

        pnl_daily   = signal * returns

        df_pnl = pd.DataFrame({
            'date'      : test.index,
            'signal'    : signal,
            'return'    : returns,
            'pnl'       : pnl_daily,
            'prob'      : prob
        })

        all_pnl.append(df_pnl)

    return pd.concat(all_pnl).set_index('date')


def compute_metrics(pnl: pd.Series) -> dict:
    """
    Calcule les métriques de performance à partir du P&L journalier.
    """
    # Rendement annualisé
    annual_return = pnl.mean() * 252

    # Volatilité annualisée
    annual_vol = pnl.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = annual_return / annual_vol

    # Courbe de valeur cumulée (1€ investi au départ)
    cumulative = (1 + pnl).cumprod()

    # Max drawdown
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    # Benchmark : buy and hold SPY (on compare à simplement garder SPY)
    return {
        'rendement_annuel' : round(annual_return * 100, 2),
        'volatilite_annuelle' : round(annual_vol * 100, 2),
        'sharpe'           : round(sharpe, 3),
        'max_drawdown'     : round(max_dd * 100, 2),
        'nb_jours'         : len(pnl)
    }


def plot_results(pnl_df: pd.DataFrame, returns_spy: pd.Series):
    """
    Trace la courbe de valeur cumulée de la stratégie vs buy and hold SPY.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # --- Graphique 1 : valeur cumulée ---
    strat_cumulative = (1 + pnl_df['pnl']).cumprod()
    spy_cumulative   = (1 + returns_spy.loc[pnl_df.index]).cumprod()

    axes[0].plot(strat_cumulative.index, strat_cumulative.values,
                 label='Stratégie ML', color='steelblue', linewidth=1.5)
    axes[0].plot(spy_cumulative.index, spy_cumulative.values,
                 label='Buy & Hold SPY', color='gray', linewidth=1.5, linestyle='--')
    axes[0].set_title('Valeur cumulée (1€ investi)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Graphique 2 : drawdown ---
    rolling_max = strat_cumulative.cummax()
    drawdown    = (strat_cumulative - rolling_max) / rolling_max * 100
    axes[1].fill_between(drawdown.index, drawdown.values, 0,
                         color='red', alpha=0.4, label='Drawdown')
    axes[1].set_title('Drawdown (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Graphique 3 : P&L journalier ---
    axes[2].bar(pnl_df.index, pnl_df['pnl'] * 100,
                color=['steelblue' if x > 0 else 'salmon' for x in pnl_df['pnl']],
                width=1, alpha=0.7)
    axes[2].set_title('P&L journalier (%)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data/backtest_results.png', dpi=150, bbox_inches='tight')
    print("Graphique sauvegardé dans data/backtest_results.png")
    plt.show()


if __name__ == '__main__':
    print("Chargement des données...")
    df   = get_data('SPY', '2018-01-01', '2024-01-01')
    feat = compute_features(df)

    print("Entraînement du modèle...")
    results, models = walk_forward_validation(feat)

    print("Calcul du P&L...")
    pnl_df = compute_pnl(feat, models)

    print("\n=== MÉTRIQUES DE LA STRATÉGIE ===")
    metrics = compute_metrics(pnl_df['pnl'])
    for k, v in metrics.items():
        print(f"  {k:25s} : {v}")

    # Buy & Hold pour comparaison
    spy_returns = feat['log_ret']
    print("\n=== MÉTRIQUES BUY & HOLD SPY ===")
    bh_metrics = compute_metrics(spy_returns.loc[pnl_df.index])
    for k, v in bh_metrics.items():
        print(f"  {k:25s} : {v}")

    print("\nGénération des graphiques...")
    plot_results(pnl_df, feat['log_ret'])