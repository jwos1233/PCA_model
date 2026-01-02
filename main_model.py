"""
Bitcoin Factor Allocation Model - Visual Dashboard
===================================================
A factor-based regime model with visual output showing:
  - BTC price with regime overlay (overweight/neutral/underweight)
  - Factor z-scores panel with current signals

Usage:
    python btc_factor_model.py

Requirements:
    pip install pandas numpy requests matplotlib

Author: Factor Research
Date: December 2025
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Model configuration parameters"""

    # Lookback periods
    ZSCORE_LOOKBACK = 30  # 3 months for z-score calculation
    MOMENTUM_LOOKBACK = 21  # 1 month for momentum signals
    VOL_LOOKBACK = 21  # 1 month for realized volatility

    # Trend filter
    EMA_PERIOD = 50  # 50-day EMA for trend filter

    # Signal thresholds (z-score based)
    OVERWEIGHT_THRESHOLD = 0.35   # Above this = overweight
    UNDERWEIGHT_THRESHOLD = -0.5  # Below this = underweight

    # Factor weights - derived from PCA variance explained (Table A1 in paper)
    # PC1: 15.45%, PC2: 11.07%, PC3: 10.62%, PC4: 9.59%, PC5: 8.90%
    # Total first 5 PCs = 55.63%, normalized weights below
    FACTOR_WEIGHTS = {
        'equity_factor': 0.278,       # PC1: Nasdaq + Momentum (15.45/55.63)
        'volatility_factor': 0.199,   # PC2: VIX + BTC Implied Vol + M2 (11.07/55.63)
        'currency_factor': 0.191,     # PC3: DXY + JPY + Gold + Basis (10.62/55.63)
        'credit_factor': 0.172,       # PC4: OAS + MOVE (9.59/55.63)
        'rates_factor': 0.160,        # PC5: Real Yields (8.90/55.63)
    }

    # Tickers to fetch
    TICKERS = {
        'btc': 'BTC-USD',
        'nasdaq': '^IXIC',
        'sp_momentum': 'SPMO',
        'vix': '^VIX',
        'usdjpy': 'JPY=X',
        'dxy': 'DX-Y.NYB',
        'gold': 'GC=F',
        'tnx': '^TNX',
        'hyg': 'HYG',
        'lqd': 'LQD',
    }

    # FRED API key for NFCI (optional)
    FRED_API_KEY = None

    # Chart colors
    COLORS = {
        'overweight': '#2ecc71',    # Green
        'neutral': '#f39c12',        # Orange/Yellow
        'underweight': '#e74c3c',    # Red
        'btc_line': '#f7931a',       # Bitcoin orange
        'background': '#1a1a2e',     # Dark background
        'text': '#ffffff',           # White text
        'grid': '#333355',           # Grid lines
    }


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_yahoo_ticker(ticker: str, days: int = 365) -> pd.Series:
    """Fetch historical data directly from Yahoo Finance API"""

    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    params = {
        'period1': start_ts,
        'period2': end_ts,
        'interval': '1d',
        'events': 'history',
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]

        if 'adjclose' in result['indicators'] and result['indicators']['adjclose'][0]['adjclose']:
            prices = result['indicators']['adjclose'][0]['adjclose']
        else:
            prices = quotes['close']

        dates = pd.to_datetime(timestamps, unit='s').normalize()
        series = pd.Series(prices, index=dates, name=ticker)
        series = series.dropna()

        return series

    except Exception as e:
        return pd.Series(dtype=float)


def fetch_all_yahoo_data(tickers: dict, days: int = 365) -> pd.DataFrame:
    """Fetch all tickers and combine into DataFrame"""

    print("Fetching market data from Yahoo Finance...")
    print("-" * 50)

    data = pd.DataFrame()

    for name, ticker in tickers.items():
        series = fetch_yahoo_ticker(ticker, days)

        if len(series) > 0:
            data[name] = series
            print(f"  ✓ {name:<15} ({ticker:<12}) - {len(series)} days")
        else:
            print(f"  ✗ {name:<15} ({ticker:<12}) - Failed")

        time.sleep(0.3)

    data = data.dropna(how='all')
    print("-" * 50)
    print(f"Total: {len(data)} trading days loaded")

    return data


def fetch_nfci_from_fred(api_key: str = None, days: int = 365) -> pd.Series:
    """Fetch NFCI directly from FRED API"""

    if api_key is None:
        api_key = os.environ.get('FRED_API_KEY')

    if api_key is None:
        return None

    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': 'NFCI',
            'api_key': api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        observations = data['observations']
        dates = [obs['date'] for obs in observations]
        values = [float(obs['value']) if obs['value'] != '.' else np.nan for obs in observations]

        series = pd.Series(values, index=pd.to_datetime(dates), name='NFCI')
        series = series.dropna()

        print(f"  ✓ NFCI (FRED) - {len(series)} observations")
        return series

    except Exception as e:
        print(f"  ✗ NFCI (FRED) - Error: {e}")
        return None


def fetch_deribit_dvol() -> pd.Series:
    """Fetch Bitcoin DVOL from Deribit"""

    try:
        url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
        params = {
            'currency': 'BTC',
            'resolution': '1D',
            'start_timestamp': int((datetime.now() - timedelta(days=365)).timestamp() * 1000),
            'end_timestamp': int(datetime.now().timestamp() * 1000),
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'result' in data and 'data' in data['result']:
            records = data['result']['data']
            dates = [datetime.fromtimestamp(r[0]/1000) for r in records]
            values = [r[4] for r in records]

            series = pd.Series(values, index=pd.to_datetime(dates), name='DVOL')
            series = series.dropna()

            print(f"  ✓ DVOL (Deribit) - {len(series)} observations")
            return series

        return None

    except Exception as e:
        print(f"  ✗ DVOL (Deribit) - Error: {e}")
        return None


# =============================================================================
# FACTOR CALCULATIONS
# =============================================================================

def calculate_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """Calculate rolling z-score"""
    mean = series.rolling(window=lookback, min_periods=lookback//2).mean()
    std = series.rolling(window=lookback, min_periods=lookback//2).std()
    zscore = (series - mean) / std
    return zscore.replace([np.inf, -np.inf], np.nan)


def calculate_hurst(series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent using R/S (Rescaled Range) method.

    H > 0.5: Trending (persistent)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting (anti-persistent)

    Returns single Hurst value for the series.
    """
    series = series.dropna()
    if len(series) < max_lag * 2:
        return 0.5  # Default to random walk if insufficient data

    lags = range(2, max_lag + 1)
    rs_values = []

    for lag in lags:
        # Split series into chunks of size 'lag'
        n_chunks = len(series) // lag
        if n_chunks < 1:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = series.iloc[i * lag:(i + 1) * lag].values

            # Mean-adjusted cumulative deviation
            mean = np.mean(chunk)
            deviations = chunk - mean
            cumulative = np.cumsum(deviations)

            # Range
            R = np.max(cumulative) - np.min(cumulative)

            # Standard deviation
            S = np.std(chunk, ddof=1)

            if S > 0:
                rs_chunk.append(R / S)

        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))

    if len(rs_values) < 3:
        return 0.5

    # Linear regression of log(R/S) vs log(lag)
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])

    # Simple linear regression for slope (Hurst exponent)
    n = len(log_lags)
    hurst = (n * np.sum(log_lags * log_rs) - np.sum(log_lags) * np.sum(log_rs)) / \
            (n * np.sum(log_lags ** 2) - np.sum(log_lags) ** 2)

    # Clip to reasonable range
    return np.clip(hurst, 0.1, 0.9)


def calculate_rolling_hurst(series: pd.Series, window: int = 63, max_lag: int = 20) -> pd.Series:
    """Calculate rolling Hurst exponent"""
    hurst_values = []

    for i in range(len(series)):
        if i < window:
            hurst_values.append(np.nan)
        else:
            window_data = series.iloc[i - window:i]
            h = calculate_hurst(window_data, max_lag)
            hurst_values.append(h)

    return pd.Series(hurst_values, index=series.index, name='hurst')


def calculate_momentum(series: pd.Series, lookback: int) -> pd.Series:
    """Calculate momentum as percentage change"""
    return series.pct_change(lookback) * 100


def calculate_realized_vol(returns: pd.Series, lookback: int) -> pd.Series:
    """Calculate annualized realized volatility"""
    return returns.rolling(window=lookback, min_periods=lookback//2).std() * np.sqrt(252) * 100


def calculate_factors(data: pd.DataFrame, dvol: pd.Series = None) -> pd.DataFrame:
    """
    Calculate all factor signals with Hurst-adjusted component confidence.

    Logic:
    - Each component gets a z-score (direction of signal)
    - Hurst exponent determines confidence: Adjusted Z = Raw Z × (H / 0.5)
    - H > 0.5 (trending) → amplify the signal (trust the trend)
    - H < 0.5 (mean-reverting) → dampen the signal (may reverse)
    - Factor z-score = average of Hurst-adjusted component z-scores
    - PCA weights applied at final composite level (structural importance)

    Rotated factor structure from paper:
    Factor 1 (Equity/Momentum): Nasdaq (0.95), Momentum (0.94)
    Factor 2 (Liquidity/Volatility): M2 (0.74), Implied Vol (-0.67), VIX (-0.42)
    Factor 3 (Currency/Carry): JPY (0.62), Basis (0.46), Gold (-0.64)
    Factor 4 (Credit Risk): OAS (0.81), MOVE (0.59), DXY (-0.42)
    Factor 5 (Real Rates): Real Yields (0.81)
    """

    factors = pd.DataFrame(index=data.index)
    cfg = Config()
    hurst_window = 63  # 3 months for Hurst calculation

    def hurst_adjusted_zscore(series: pd.Series, name: str, invert: bool = False) -> tuple:
        """
        Calculate z-score adjusted by Hurst confidence.
        Returns: (adjusted_z_series, hurst_series, raw_z_series)
        """
        if series is None or len(series.dropna()) < hurst_window:
            return None, None, None

        # Calculate momentum/change
        momentum = calculate_momentum(series, cfg.MOMENTUM_LOOKBACK)
        if invert:
            momentum = -momentum

        # Raw z-score
        raw_z = calculate_zscore(momentum, cfg.ZSCORE_LOOKBACK)

        # Rolling Hurst
        hurst = calculate_rolling_hurst(raw_z.dropna(), window=hurst_window)
        hurst = hurst.reindex(raw_z.index, method='ffill')

        # Confidence multiplier: H/0.5, clipped to 0.5-1.5x
        confidence = (hurst / 0.5).clip(0.5, 1.5)

        # Adjusted z-score
        adjusted_z = raw_z * confidence

        return adjusted_z, hurst, raw_z

    # =========================================================================
    # FACTOR 1: EQUITY/MOMENTUM (PC1 - 27.8% weight)
    # Components: Nasdaq (0.95), Momentum (0.94) - both bullish when up
    # =========================================================================
    equity_components = []

    if 'nasdaq' in data.columns:
        adj_z, h, raw_z = hurst_adjusted_zscore(data['nasdaq'], 'nasdaq')
        if adj_z is not None:
            factors['nasdaq_z'] = raw_z
            factors['nasdaq_hurst'] = h
            factors['nasdaq_adj_z'] = adj_z
            equity_components.append('nasdaq_adj_z')

    if 'sp_momentum' in data.columns:
        adj_z, h, raw_z = hurst_adjusted_zscore(data['sp_momentum'], 'sp_momentum')
        if adj_z is not None:
            factors['momentum_z'] = raw_z
            factors['momentum_hurst'] = h
            factors['momentum_adj_z'] = adj_z
            equity_components.append('momentum_adj_z')

    if equity_components:
        factors['equity_factor'] = factors[equity_components].mean(axis=1)

    # =========================================================================
    # FACTOR 2: VOLATILITY/LIQUIDITY (PC2 - 19.9% weight)
    # Components: VIX (-0.42), Implied Vol (-0.67) - bullish when DOWN
    # =========================================================================
    vol_components = []

    if 'vix' in data.columns:
        adj_z, h, raw_z = hurst_adjusted_zscore(data['vix'], 'vix', invert=True)
        if adj_z is not None:
            factors['vix_z'] = raw_z
            factors['vix_hurst'] = h
            factors['vix_adj_z'] = adj_z
            vol_components.append('vix_adj_z')

    # Use DVOL if available, otherwise BTC realized vol
    if dvol is not None and len(dvol) > 0:
        adj_z, h, raw_z = hurst_adjusted_zscore(dvol, 'dvol', invert=True)
        if adj_z is not None:
            factors['btcvol_z'] = raw_z
            factors['btcvol_hurst'] = h
            factors['btcvol_adj_z'] = adj_z
            vol_components.append('btcvol_adj_z')
    elif 'btc' in data.columns:
        btc_returns = data['btc'].pct_change()
        btc_rvol = calculate_realized_vol(btc_returns, cfg.VOL_LOOKBACK)
        if len(btc_rvol.dropna()) > hurst_window:
            raw_z = calculate_zscore(-btc_rvol, cfg.ZSCORE_LOOKBACK)  # Inverted
            hurst = calculate_rolling_hurst(raw_z.dropna(), window=hurst_window)
            hurst = hurst.reindex(raw_z.index, method='ffill')
            confidence = (hurst / 0.5).clip(0.5, 1.5)
            factors['btcvol_z'] = raw_z
            factors['btcvol_hurst'] = hurst
            factors['btcvol_adj_z'] = raw_z * confidence
            vol_components.append('btcvol_adj_z')

    if vol_components:
        factors['volatility_factor'] = factors[vol_components].mean(axis=1)

    # =========================================================================
    # FACTOR 3: CURRENCY/CARRY (PC3 - 19.1% weight)
    # Components: JPY (+0.62) bullish when up, Gold (-0.64) bullish when down
    # =========================================================================
    currency_components = []

    if 'usdjpy' in data.columns:
        adj_z, h, raw_z = hurst_adjusted_zscore(data['usdjpy'], 'usdjpy', invert=False)
        if adj_z is not None:
            factors['jpy_z'] = raw_z
            factors['jpy_hurst'] = h
            factors['jpy_adj_z'] = adj_z
            currency_components.append('jpy_adj_z')

    if 'gold' in data.columns:
        adj_z, h, raw_z = hurst_adjusted_zscore(data['gold'], 'gold', invert=False)  # Gold up = bullish
        if adj_z is not None:
            factors['gold_z'] = raw_z
            factors['gold_hurst'] = h
            factors['gold_adj_z'] = adj_z
            currency_components.append('gold_adj_z')

    if 'dxy' in data.columns:
        adj_z, h, raw_z = hurst_adjusted_zscore(data['dxy'], 'dxy', invert=True)
        if adj_z is not None:
            factors['dxy_z'] = raw_z
            factors['dxy_hurst'] = h
            factors['dxy_adj_z'] = adj_z
            currency_components.append('dxy_adj_z')

    if currency_components:
        factors['currency_factor'] = factors[currency_components].mean(axis=1)

    # =========================================================================
    # FACTOR 4: CREDIT RISK (PC4 - 17.2% weight)
    # Components: OAS (+0.81) - tighter spreads = bullish, so HYG outperformance = bullish
    # =========================================================================
    credit_components = []

    if 'hyg' in data.columns and 'lqd' in data.columns:
        # HYG-LQD spread: positive = tighter spreads = bullish
        spread = data['hyg'] / data['lqd']  # Ratio approach
        adj_z, h, raw_z = hurst_adjusted_zscore(spread, 'credit', invert=False)
        if adj_z is not None:
            factors['credit_z'] = raw_z
            factors['credit_hurst'] = h
            factors['credit_adj_z'] = adj_z
            credit_components.append('credit_adj_z')

    if credit_components:
        factors['credit_factor'] = factors[credit_components].mean(axis=1)

    # =========================================================================
    # FACTOR 5: REAL RATES (PC5 - 16.0% weight)
    # Components: Real Yields (+0.81) - lower yields = bullish
    # =========================================================================
    if 'tnx' in data.columns:
        adj_z, h, raw_z = hurst_adjusted_zscore(data['tnx'], 'rates', invert=True)
        if adj_z is not None:
            factors['rates_z'] = raw_z
            factors['rates_hurst'] = h
            factors['rates_adj_z'] = adj_z
            factors['rates_factor'] = adj_z

    return factors


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_composite_signal(factors: pd.DataFrame, btc_prices: pd.Series = None) -> pd.DataFrame:
    """
    Generate weighted composite signal using fixed PCA weights.

    Includes 50-day EMA trend filter:
    - If BTC price < 50 EMA → cannot be OVERWEIGHT (capped at NEUTRAL)

    Hurst adjustment is already applied at the component level within each factor.
    Here we just apply the structural PCA weights to combine factors.
    """

    cfg = Config()
    signals = pd.DataFrame(index=factors.index)

    # Fixed PCA weights (structural importance)
    pca_factors = ['equity_factor', 'volatility_factor', 'currency_factor',
                   'credit_factor', 'rates_factor']

    # Check which factors are available
    available_factors = {}
    total_weight = 0

    for factor_name in pca_factors:
        if factor_name in cfg.FACTOR_WEIGHTS:
            weight = cfg.FACTOR_WEIGHTS[factor_name]
            if factor_name in factors.columns and factors[factor_name].notna().any():
                available_factors[factor_name] = weight
                total_weight += weight

    # Normalize weights
    if total_weight > 0:
        for factor_name in available_factors:
            available_factors[factor_name] /= total_weight

    # Calculate composite (Hurst adjustment already in factor values)
    signals['composite_zscore'] = 0.0
    for factor_name, weight in available_factors.items():
        signals['composite_zscore'] += factors[factor_name].fillna(0) * weight

    # Copy component-level Hurst values for display
    hurst_cols = [c for c in factors.columns if c.endswith('_hurst')]
    for col in hurst_cols:
        signals[col] = factors[col]

    # Copy raw z-scores for display
    z_cols = [c for c in factors.columns if c.endswith('_z') and not c.endswith('_adj_z')]
    for col in z_cols:
        signals[col] = factors[col]

    # =========================================================================
    # TREND FILTER: 50-day EMA
    # =========================================================================
    if btc_prices is not None:
        btc_aligned = btc_prices.reindex(signals.index, method='ffill')
        signals['btc_price'] = btc_aligned
        signals['ema_50'] = btc_aligned.ewm(span=cfg.EMA_PERIOD, adjust=False).mean()
        signals['above_ema'] = btc_aligned > signals['ema_50']
    else:
        signals['above_ema'] = True  # No filter if no BTC prices provided

    # Generate raw signal (before trend filter)
    def classify_raw(z):
        if pd.isna(z):
            return 'NEUTRAL'
        elif z > cfg.OVERWEIGHT_THRESHOLD:
            return 'OVERWEIGHT'
        elif z < cfg.UNDERWEIGHT_THRESHOLD:
            return 'UNDERWEIGHT'
        else:
            return 'NEUTRAL'

    signals['signal_raw'] = signals['composite_zscore'].apply(classify_raw)

    # Apply trend filter: can only be OVERWEIGHT if above EMA
    def apply_trend_filter(row):
        raw_signal = row['signal_raw']
        above_ema = row['above_ema']

        if raw_signal == 'OVERWEIGHT' and not above_ema:
            return 'NEUTRAL'  # Cap at neutral when below EMA
        return raw_signal

    signals['signal'] = signals.apply(apply_trend_filter, axis=1)
    signals['trend_filtered'] = (signals['signal_raw'] == 'OVERWEIGHT') & (~signals['above_ema'])
    signals['conviction'] = signals['composite_zscore'].abs().clip(upper=2) / 2 * 100

    return signals


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_dashboard(data: pd.DataFrame, factors: pd.DataFrame, signals: pd.DataFrame,
                     save_path: str = 'btc_factor_dashboard.png'):
    """Create comprehensive visual dashboard"""

    cfg = Config()

    # Setup figure with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14))

    # Define grid: 3 rows
    # Row 1: BTC price with regime overlay (larger)
    # Row 2: Composite signal z-score
    # Row 3: Factor regime heatmap

    gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 1, 1.5], hspace=0.3)

    # Colors
    colors = cfg.COLORS

    # Get aligned data
    btc = data['btc'].reindex(signals.index)

    # =========================================================================
    # PANEL 1: BTC Price with Regime Overlay
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])

    # Plot regime backgrounds
    signal_colors = {
        'OVERWEIGHT': colors['overweight'],
        'NEUTRAL': colors['neutral'],
        'UNDERWEIGHT': colors['underweight']
    }

    # Create regime spans
    current_regime = None
    regime_start = None

    for i, (date, row) in enumerate(signals.iterrows()):
        if row['signal'] != current_regime:
            if current_regime is not None:
                ax1.axvspan(regime_start, date, alpha=0.2,
                           color=signal_colors.get(current_regime, 'gray'),
                           linewidth=0)
            current_regime = row['signal']
            regime_start = date

    # Final regime
    if current_regime is not None:
        ax1.axvspan(regime_start, signals.index[-1], alpha=0.2,
                   color=signal_colors.get(current_regime, 'gray'),
                   linewidth=0)

    # Plot BTC price
    ax1.plot(btc.index, btc.values, color=colors['btc_line'], linewidth=2, label='BTC Price')

    # Plot 50 EMA
    if 'ema_50' in signals.columns:
        ema_50 = signals['ema_50']
        ax1.plot(ema_50.index, ema_50.values, color='#00ffff', linewidth=1.5,
                linestyle='--', alpha=0.8, label='50 EMA')

    # Formatting
    ax1.set_ylabel('BTC Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    ax1.set_xlim(signals.index[0], signals.index[-1])

    # Current price annotation
    current_price = btc.iloc[-1]
    ax1.annotate(f'${current_price:,.0f}',
                xy=(btc.index[-1], current_price),
                xytext=(10, 0), textcoords='offset points',
                fontsize=11, fontweight='bold', color=colors['btc_line'],
                va='center')

    # Title with current signal and trend filter status
    current_signal = signals['signal'].iloc[-1]
    current_zscore = signals['composite_zscore'].iloc[-1]
    signal_emoji = {'OVERWEIGHT': '🟢', 'NEUTRAL': '🟡', 'UNDERWEIGHT': '🔴'}

    # Check if trend filter is currently active
    trend_filter_active = signals['trend_filtered'].iloc[-1] if 'trend_filtered' in signals.columns else False
    above_ema = signals['above_ema'].iloc[-1] if 'above_ema' in signals.columns else True

    title = f"Bitcoin Factor Allocation Model  |  Current Signal: {signal_emoji.get(current_signal, '')} {current_signal} (z={current_zscore:+.2f})"
    if trend_filter_active:
        title += "  ⚠️ [TREND FILTERED]"
    elif not above_ema:
        title += "  📉 [Below 50 EMA]"
    else:
        title += "  📈 [Above 50 EMA]"
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Legend for regimes
    legend_patches = [
        mpatches.Patch(color=colors['overweight'], alpha=0.3, label='Overweight'),
        mpatches.Patch(color=colors['neutral'], alpha=0.3, label='Neutral'),
        mpatches.Patch(color=colors['underweight'], alpha=0.3, label='Underweight'),
        plt.Line2D([0], [0], color='#00ffff', linestyle='--', linewidth=1.5, label='50 EMA'),
    ]
    ax1.legend(handles=legend_patches, loc='upper left', fontsize=10)

    # =========================================================================
    # PANEL 2: Composite Z-Score
    # =========================================================================
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot composite z-score as area
    zscore = signals['composite_zscore']

    # Fill areas based on signal
    ax2.fill_between(zscore.index, 0, zscore.values,
                     where=(zscore > cfg.OVERWEIGHT_THRESHOLD),
                     color=colors['overweight'], alpha=0.6, label='Overweight')
    ax2.fill_between(zscore.index, 0, zscore.values,
                     where=(zscore < cfg.UNDERWEIGHT_THRESHOLD),
                     color=colors['underweight'], alpha=0.6, label='Underweight')
    ax2.fill_between(zscore.index, 0, zscore.values,
                     where=((zscore >= cfg.UNDERWEIGHT_THRESHOLD) & (zscore <= cfg.OVERWEIGHT_THRESHOLD)),
                     color=colors['neutral'], alpha=0.6, label='Neutral')

    # Threshold lines
    ax2.axhline(y=cfg.OVERWEIGHT_THRESHOLD, color=colors['overweight'], linestyle='--', alpha=0.7, linewidth=1)
    ax2.axhline(y=cfg.UNDERWEIGHT_THRESHOLD, color=colors['underweight'], linestyle='--', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=0.5)

    # Current value annotation
    ax2.annotate(f'{current_zscore:+.2f}',
                xy=(zscore.index[-1], current_zscore),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold',
                color=signal_colors.get(current_signal, 'white'),
                va='center')

    ax2.set_ylabel('Composite\nZ-Score', fontsize=11, fontweight='bold')
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, alpha=0.3, color=colors['grid'])

    # =========================================================================
    # PANEL 3: Factor Regime Heatmap (rows = factors, columns = days)
    # =========================================================================
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # 5 factors from PCA with their variance weights
    factor_info = {
        'equity_factor': ('Equity/Momo (27.8%)', '#3498db'),
        'volatility_factor': ('Vol/Liquidity (19.9%)', '#9b59b6'),
        'currency_factor': ('Currency/Carry (19.1%)', '#1abc9c'),
        'credit_factor': ('Credit Risk (17.2%)', '#e74c3c'),
        'rates_factor': ('Real Rates (16.0%)', '#f39c12'),
    }

    # Build regime matrix: rows = factors, columns = dates
    factor_keys = list(factor_info.keys())
    n_factors = len(factor_keys)
    n_days = len(factors)

    # Create numeric regime matrix: 1 = bullish, 0 = neutral, -1 = bearish
    regime_matrix = np.zeros((n_factors, n_days))

    for i, factor_key in enumerate(factor_keys):
        if factor_key in factors.columns:
            z_values = factors[factor_key].values
            regime_matrix[i, :] = np.where(
                z_values > cfg.OVERWEIGHT_THRESHOLD, 1,
                np.where(z_values < cfg.UNDERWEIGHT_THRESHOLD, -1, 0)
            )

    # Custom colormap: red (-1) -> orange (0) -> green (1)
    from matplotlib.colors import ListedColormap
    regime_cmap = ListedColormap([colors['underweight'], colors['neutral'], colors['overweight']])

    # Plot heatmap
    im = ax3.imshow(
        regime_matrix,
        aspect='auto',
        cmap=regime_cmap,
        vmin=-1, vmax=1,
        extent=[mdates.date2num(factors.index[0]), mdates.date2num(factors.index[-1]),
                n_factors - 0.5, -0.5],
        interpolation='nearest'
    )

    # Get current Hurst values for each factor (average of components) and color labels
    latest_factors_data = factors.iloc[-1]
    factor_labels = []
    label_colors = []

    component_map = {
        'equity_factor': ['nasdaq_hurst', 'momentum_hurst'],
        'volatility_factor': ['vix_hurst', 'btcvol_hurst'],
        'currency_factor': ['jpy_hurst', 'gold_hurst', 'dxy_hurst'],
        'credit_factor': ['credit_hurst'],
        'rates_factor': ['rates_hurst'],
    }

    for factor_key in factor_keys:
        base_label = factor_info[factor_key][0]

        # Get average Hurst of components
        hurst_cols = component_map.get(factor_key, [])
        hurst_values = []
        for hc in hurst_cols:
            if hc in factors.columns:
                h = latest_factors_data[hc]
                if pd.notna(h):
                    hurst_values.append(h)

        if hurst_values:
            avg_h = np.mean(hurst_values)
            # Color by average Hurst
            if avg_h > 0.55:
                hurst_color = colors['overweight']  # Trending - green
                trend_indicator = "↗"
            elif avg_h < 0.45:
                hurst_color = colors['underweight']  # Mean-reverting - red
                trend_indicator = "↺"
            else:
                hurst_color = colors['neutral']  # Random walk - orange
                trend_indicator = "→"
            factor_labels.append(f"{base_label} H:{avg_h:.2f}{trend_indicator}")
            label_colors.append(hurst_color)
        else:
            factor_labels.append(base_label)
            label_colors.append('white')

    # Set y-axis labels with Hurst-based colors
    ax3.set_yticks(range(n_factors))
    y_labels = ax3.set_yticklabels(factor_labels, fontsize=9)

    # Color each label
    for label, color in zip(y_labels, label_colors):
        label.set_color(color)

    # Format x-axis
    ax3.xaxis_date()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Factor Regimes (labels colored by Hurst: ↗trending  →random  ↺reverting)',
                  fontsize=10, fontweight='bold', pad=10)

    # Add grid lines between factors
    for i in range(n_factors - 1):
        ax3.axhline(y=i + 0.5, color='white', linewidth=0.5, alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['overweight'], label='Bullish (z > 0.5)'),
        mpatches.Patch(facecolor=colors['neutral'], label='Neutral'),
        mpatches.Patch(facecolor=colors['underweight'], label='Bearish (z < -0.5)'),
    ]
    ax3.legend(handles=legend_elements, loc='upper left', ncol=3, fontsize=8, framealpha=0.8)

    # Rotate x-axis labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =========================================================================
    # Add Current Factor Values Panel (right side annotation)
    # =========================================================================
    latest_factors = factors.iloc[-1]
    latest_signals = signals.iloc[-1]

    cfg = Config()

    # Component details for each factor
    component_map = {
        'equity_factor': [('nasdaq', 'Nasdaq'), ('momentum', 'Momentum')],
        'volatility_factor': [('vix', 'VIX'), ('btcvol', 'BTC Vol')],
        'currency_factor': [('jpy', 'USD/JPY'), ('gold', 'Gold'), ('dxy', 'DXY')],
        'credit_factor': [('credit', 'HYG/LQD')],
        'rates_factor': [('rates', '10Y Yield')],
    }

    factor_text = "COMPONENT BREAKDOWN\n" + "─" * 32 + "\n"
    factor_text += f"{'Component':<12} {'Z':>5} {'H':>5} {'Adj':>5} {'Conf':<6}\n"
    factor_text += "─" * 32 + "\n"

    for factor_key in factor_info.keys():
        if factor_key in component_map:
            # Factor header
            factor_label = factor_info[factor_key][0].split('(')[0].strip()
            factor_text += f"\n{factor_label}\n"

            for comp_key, comp_name in component_map[factor_key]:
                z_col = f'{comp_key}_z'
                h_col = f'{comp_key}_hurst'
                adj_col = f'{comp_key}_adj_z'

                if z_col in factors.columns:
                    z = latest_factors[z_col] if z_col in latest_factors else np.nan
                    h = latest_factors[h_col] if h_col in latest_factors else np.nan
                    adj = latest_factors[adj_col] if adj_col in latest_factors else np.nan

                    if pd.notna(z) and pd.notna(h):
                        # Confidence indicator
                        if h > 0.55:
                            conf = "↗ High"
                        elif h < 0.45:
                            conf = "↺ Low"
                        else:
                            conf = "→ Med"

                        factor_text += f"  {comp_name:<10} {z:>+.1f} {h:>.2f} {adj:>+.1f} {conf}\n"

    factor_text += "\n" + "─" * 32 + "\n"
    factor_text += "H>0.5: Trending (amplify)\n"
    factor_text += "H<0.5: Reverting (dampen)\n"
    factor_text += "Adj = Z × (H/0.5)"

    # Add text box
    props = dict(boxstyle='round,pad=0.5', facecolor='#2d2d44', alpha=0.9, edgecolor='#555577')
    ax3.text(1.02, 0.98, factor_text, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=props, color='white')

    # =========================================================================
    # Footer
    # =========================================================================
    fig.text(0.5, 0.01,
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
             f"Data through: {signals.index[-1].strftime('%Y-%m-%d')}  |  "
             f"Component signals adjusted by Hurst confidence (H/0.5)",
             ha='center', fontsize=9, color='gray', style='italic')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, right=0.85, left=0.15)

    # Save
    plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
    print(f"\n✓ Dashboard saved to: {save_path}")

    # Also display
    plt.show()

    return fig


def print_current_signal(signals: pd.DataFrame, factors: pd.DataFrame):
    """Print current signal summary to console"""

    cfg = Config()
    latest = signals.iloc[-1]
    latest_factors = factors.iloc[-1]

    signal_emoji = {'OVERWEIGHT': '🟢', 'NEUTRAL': '🟡', 'UNDERWEIGHT': '🔴'}

    # Trend filter info
    above_ema = latest['above_ema'] if 'above_ema' in latest else True
    trend_filtered = latest['trend_filtered'] if 'trend_filtered' in latest else False
    raw_signal = latest['signal_raw'] if 'signal_raw' in latest else latest['signal']

    print("\n" + "=" * 75)
    print("  CURRENT SIGNAL SUMMARY")
    print("=" * 75)
    print(f"""
    {signal_emoji.get(latest['signal'], '')}  Signal:      {latest['signal']}
    Z-Score:     {latest['composite_zscore']:+.2f}
    Conviction:  {latest['conviction']:.0f}%
    Date:        {signals.index[-1].strftime('%Y-%m-%d')}
    """)

    # Trend filter status
    print("  50-day EMA Trend Filter:")
    print("  " + "-" * 70)
    if 'btc_price' in latest and 'ema_50' in latest:
        btc_price = latest['btc_price']
        ema_50 = latest['ema_50']
        pct_from_ema = ((btc_price / ema_50) - 1) * 100
        ema_status = "📈 ABOVE" if above_ema else "📉 BELOW"
        print(f"    BTC Price:   ${btc_price:,.0f}")
        print(f"    50 EMA:      ${ema_50:,.0f}")
        print(f"    Status:      {ema_status} ({pct_from_ema:+.1f}% from EMA)")
    if trend_filtered:
        print(f"    ⚠️  FILTER ACTIVE: Raw signal was {raw_signal}, capped to NEUTRAL")
    print()

    print("  Component Breakdown (Hurst-adjusted signals):")
    print("  " + "-" * 70)
    print(f"    {'Factor':<18} {'Component':<12} {'Raw Z':>7} {'Hurst':>7} {'Adj Z':>7} {'Conf':>8}")
    print("  " + "-" * 70)

    # Component details for each factor
    component_map = {
        'equity_factor': [('nasdaq', 'Nasdaq'), ('momentum', 'Momentum')],
        'volatility_factor': [('vix', 'VIX'), ('btcvol', 'BTC Vol')],
        'currency_factor': [('jpy', 'USD/JPY'), ('gold', 'Gold'), ('dxy', 'DXY')],
        'credit_factor': [('credit', 'HYG/LQD')],
        'rates_factor': [('rates', '10Y Yield')],
    }

    factor_names = {
        'equity_factor': 'Equity/Momo (27.8%)',
        'volatility_factor': 'Vol/Liquidity (19.9%)',
        'currency_factor': 'Currency/Carry (19.1%)',
        'credit_factor': 'Credit Risk (17.2%)',
        'rates_factor': 'Real Rates (16.0%)',
    }

    for factor_key, factor_name in factor_names.items():
        if factor_key in component_map:
            first_component = True
            for comp_key, comp_name in component_map[factor_key]:
                z_col = f'{comp_key}_z'
                h_col = f'{comp_key}_hurst'
                adj_col = f'{comp_key}_adj_z'

                if z_col in factors.columns:
                    z = latest_factors[z_col] if z_col in latest_factors.index else np.nan
                    h = latest_factors[h_col] if h_col in latest_factors.index else np.nan
                    adj = latest_factors[adj_col] if adj_col in latest_factors.index else np.nan

                    if pd.notna(z) and pd.notna(h):
                        # Confidence indicator
                        if h > 0.55:
                            conf = "↗ Trend"
                        elif h < 0.45:
                            conf = "↺ Revert"
                        else:
                            conf = "→ Random"

                        # Show factor name only on first component
                        if first_component:
                            print(f"    {factor_name:<18} {comp_name:<12} {z:>+6.2f} {h:>7.2f} {adj:>+6.2f} {conf:>8}")
                            first_component = False
                        else:
                            print(f"    {'':<18} {comp_name:<12} {z:>+6.2f} {h:>7.2f} {adj:>+6.2f} {conf:>8}")

            # Show factor total
            if factor_key in factors.columns:
                factor_val = latest_factors[factor_key]
                if pd.notna(factor_val):
                    if factor_val > cfg.OVERWEIGHT_THRESHOLD:
                        sig = "▲ Bullish"
                    elif factor_val < cfg.UNDERWEIGHT_THRESHOLD:
                        sig = "▼ Bearish"
                    else:
                        sig = "● Neutral"
                    print(f"    {'':<18} {'→ FACTOR':<12} {'':<7} {'':<7} {factor_val:>+6.2f} {sig:>8}")
            print()

    print("  " + "-" * 70)
    print("  Formula: Adjusted Z = Raw Z × (Hurst / 0.5)")
    print("  Trending (H>0.5) amplifies signal, Mean-reverting (H<0.5) dampens it")
    print("=" * 75)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_model(save_path: str = 'btc_factor_dashboard.png'):
    """Run the full factor model pipeline"""

    print("\n" + "=" * 60)
    print("     BITCOIN FACTOR ALLOCATION MODEL")
    print("     Based on PCA Factor Analysis (5 Factors)")
    print("=" * 60)

    cfg = Config()

    # 1. Fetch Data
    print("\n[1/4] FETCHING DATA")
    print("-" * 60)

    data = fetch_all_yahoo_data(cfg.TICKERS, days=450)

    if len(data) < cfg.ZSCORE_LOOKBACK:
        print(f"\nERROR: Insufficient data. Need at least {cfg.ZSCORE_LOOKBACK} days.")
        return None, None, None

    # Fetch DVOL for Factor 2 (Volatility/Liquidity)
    print("\nFetching DVOL from Deribit...")
    dvol = fetch_deribit_dvol()
    if dvol is None:
        print("  ⚠ Using BTC realized volatility as proxy")

    # 2. Calculate Factors
    print("\n[2/4] CALCULATING PCA FACTORS")
    print("-" * 60)
    print("  Factor weights from PCA variance explained:")
    print("    PC1: Equity/Momentum    27.8%")
    print("    PC2: Volatility/Liq     19.9%")
    print("    PC3: Currency/Carry     19.1%")
    print("    PC4: Credit Risk        17.2%")
    print("    PC5: Real Rates         16.0%")
    print()

    factors = calculate_factors(data, dvol=dvol)

    available = [f for f in cfg.FACTOR_WEIGHTS.keys() if f in factors.columns and factors[f].notna().any()]
    print(f"  Active factors: {len(available)}/{len(cfg.FACTOR_WEIGHTS)}")
    for f in available:
        print(f"    ✓ {f}")

    # 3. Generate Signals
    print("\n[3/4] GENERATING SIGNALS")
    print("-" * 60)
    print("  Hurst adjustment applied at component level:")
    print("    - Each component's z-score scaled by (H/0.5)")
    print("    - H > 0.5 (trending) → amplify signal confidence")
    print("    - H < 0.5 (reverting) → dampen signal confidence")
    print("    - PCA weights remain fixed (structural importance)")
    print()
    print("  Trend Filter: 50-day EMA")
    print("    - If BTC < 50 EMA → cannot be OVERWEIGHT (capped at NEUTRAL)")
    print()

    btc_prices = data['btc'] if 'btc' in data.columns else None
    signals = generate_composite_signal(factors, btc_prices)

    signal_counts = signals['signal'].value_counts()
    print(f"  Signal distribution (after trend filter):")
    for sig, count in signal_counts.items():
        print(f"    {sig}: {count} days ({count/len(signals)*100:.1f}%)")

    # Show how many days were filtered
    if 'trend_filtered' in signals.columns:
        filtered_days = signals['trend_filtered'].sum()
        if filtered_days > 0:
            print(f"\n  ⚠ {filtered_days} days filtered from OVERWEIGHT → NEUTRAL (below EMA)")

    # Print current signal
    print_current_signal(signals, factors)

    # 4. Create Dashboard
    print("\n[4/4] CREATING DASHBOARD")
    print("-" * 60)

    fig = create_dashboard(data, factors, signals, save_path)

    return data, factors, signals


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    data, factors, signals = run_model(save_path='btc_factor_dashboard.png')

    if signals is not None:
        current = signals['signal'].iloc[-1]
        zscore = signals['composite_zscore'].iloc[-1]
        print(f"\n>>> FINAL: {current} (z={zscore:+.2f})")
