import numpy as np
import pandas as pd
import yfinance as yf


def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les données OHLCV depuis Yahoo Finance.
    ticker : 'SPY', 'AAPL', 'BNP.PA'...
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0).str.lower()
    return df[['open', 'high', 'low', 'close', 'volume']].dropna()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule tous les features à partir des données OHLCV.
    Retourne un DataFrame prêt pour le modèle ML.
    """
    feat = pd.DataFrame(index=df.index)

    # Log-returns journaliers
    feat['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # Momentum multi-échelle
    for w in [5, 21, 63]:
        feat[f'mom_{w}d'] = np.log(df['close'] / df['close'].shift(w))

    # Momentum classique (2 mois vs 12 mois)
    feat['mom_2_12'] = np.log(df['close'].shift(42) / df['close'].shift(252))

    # Volatilité réalisée annualisée
    for w in [5, 21, 63]:
        feat[f'vol_{w}d'] = feat['log_ret'].rolling(w).std() * np.sqrt(252)

    # Z-score du prix (signal mean-reversion)
    for w in [5, 21, 63]:
        roll_mean = df['close'].rolling(w).mean()
        roll_std  = df['close'].rolling(w).std()
        feat[f'zscore_{w}d'] = (df['close'] - roll_mean) / roll_std

    # RSI 14 jours
    def rsi(series, n=14):
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(n).mean()
        loss  = (-delta.clip(upper=0)).rolling(n).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    feat['rsi_14'] = rsi(df['close'])

    # Volume anormal
    feat['vol_ratio'] = df['volume'] / df['volume'].rolling(21).mean()

    # Target : direction du lendemain (1 = hausse, 0 = baisse)
    feat['target'] = (feat['log_ret'].shift(-1) > 0).astype(int)

    return feat.dropna()


# Test rapide
if __name__ == '__main__':
    print("Téléchargement des données SPY...")
    df = get_data('SPY', '2018-01-01', '2024-01-01')
    print(f"Données : {len(df)} jours")

    print("\nCalcul des features...")
    feat = compute_features(df)
    print(f"Features : {len(feat)} lignes, {len(feat.columns)} colonnes")
    print("\nPremières lignes :")
    print(feat.head())