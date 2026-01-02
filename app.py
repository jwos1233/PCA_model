"""
Bitcoin Factor Allocation Model - Web API
==========================================
FastAPI backend for serving the BTC factor model via REST API.
"""

import io
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# Import model functions
from main_model import (
    Config,
    fetch_all_yahoo_data,
    fetch_deribit_dvol,
    calculate_factors,
    generate_composite_signal,
)

app = FastAPI(
    title="Bitcoin Factor Allocation Model",
    description="A PCA-based factor model for Bitcoin allocation signals",
    version="1.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


def safe_float(value) -> Optional[float]:
    """Convert value to float, handling NaN and None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_str(value) -> Optional[str]:
    """Convert value to string, handling None."""
    if value is None:
        return None
    return str(value)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/signal")
async def get_signal():
    """
    Get current Bitcoin factor allocation signal.

    Returns the current signal (OVERWEIGHT/NEUTRAL/UNDERWEIGHT),
    composite z-score, factor breakdown, and component details.
    """
    try:
        cfg = Config()

        # Fetch market data
        data = fetch_all_yahoo_data(cfg.TICKERS, days=450)

        if len(data) < cfg.ZSCORE_LOOKBACK:
            raise HTTPException(
                status_code=500,
                detail=f"Insufficient data. Need at least {cfg.ZSCORE_LOOKBACK} days."
            )

        # Fetch DVOL
        dvol = fetch_deribit_dvol()

        # Calculate factors
        factors = calculate_factors(data, dvol=dvol)

        # Generate signals
        btc_prices = data['btc'] if 'btc' in data.columns else None
        signals = generate_composite_signal(factors, btc_prices)

        # Get latest values
        latest_signals = signals.iloc[-1]
        latest_factors = factors.iloc[-1]

        # Build factor breakdown
        factor_info = {
            'equity_factor': {'name': 'Equity/Momentum', 'weight': 27.8},
            'volatility_factor': {'name': 'Volatility/Liquidity', 'weight': 19.9},
            'currency_factor': {'name': 'Currency/Carry', 'weight': 19.1},
            'credit_factor': {'name': 'Credit Risk', 'weight': 17.2},
            'rates_factor': {'name': 'Real Rates', 'weight': 16.0},
        }

        component_map = {
            'equity_factor': [('nasdaq', 'Nasdaq'), ('momentum', 'S&P Momentum')],
            'volatility_factor': [('vix', 'VIX'), ('btcvol', 'BTC Volatility')],
            'currency_factor': [('jpy', 'USD/JPY'), ('gold', 'Gold'), ('dxy', 'Dollar Index')],
            'credit_factor': [('credit', 'HYG/LQD Spread')],
            'rates_factor': [('rates', '10Y Treasury')],
        }

        factor_breakdown = []
        for factor_key, info in factor_info.items():
            factor_value = safe_float(latest_factors.get(factor_key))

            # Determine factor signal
            if factor_value is not None:
                if factor_value > cfg.OVERWEIGHT_THRESHOLD:
                    factor_signal = "BULLISH"
                elif factor_value < cfg.UNDERWEIGHT_THRESHOLD:
                    factor_signal = "BEARISH"
                else:
                    factor_signal = "NEUTRAL"
            else:
                factor_signal = "N/A"

            # Get components
            components = []
            for comp_key, comp_name in component_map.get(factor_key, []):
                z_col = f'{comp_key}_z'
                h_col = f'{comp_key}_hurst'
                adj_col = f'{comp_key}_adj_z'

                z = safe_float(latest_factors.get(z_col))
                h = safe_float(latest_factors.get(h_col))
                adj = safe_float(latest_factors.get(adj_col))

                # Determine trend type from Hurst
                if h is not None:
                    if h > 0.55:
                        trend = "TRENDING"
                    elif h < 0.45:
                        trend = "REVERTING"
                    else:
                        trend = "RANDOM"
                else:
                    trend = "N/A"

                components.append({
                    'name': comp_name,
                    'z_score': z,
                    'hurst': h,
                    'adjusted_z': adj,
                    'trend': trend
                })

            factor_breakdown.append({
                'id': factor_key,
                'name': info['name'],
                'weight': info['weight'],
                'value': factor_value,
                'signal': factor_signal,
                'components': components
            })

        # Build historical data for charts (last 90 days)
        chart_days = 90
        history_signals = signals.tail(chart_days)
        history_factors = factors.tail(chart_days)

        price_history = []
        zscore_history = []

        for idx, row in history_signals.iterrows():
            date_str = idx.strftime('%Y-%m-%d')

            price_history.append({
                'date': date_str,
                'price': safe_float(row.get('btc_price')),
                'ema50': safe_float(row.get('ema_50')),
                'signal': safe_str(row.get('signal'))
            })

            zscore_history.append({
                'date': date_str,
                'zscore': safe_float(row.get('composite_zscore')),
                'signal': safe_str(row.get('signal'))
            })

        # Build response
        response = {
            'timestamp': datetime.now().isoformat(),
            'data_date': signals.index[-1].strftime('%Y-%m-%d'),
            'signal': {
                'current': safe_str(latest_signals.get('signal')),
                'raw': safe_str(latest_signals.get('signal_raw')),
                'z_score': safe_float(latest_signals['composite_zscore']),
                'conviction': safe_float(latest_signals.get('conviction')),
            },
            'trend_filter': {
                'btc_price': safe_float(latest_signals.get('btc_price')),
                'ema_50': safe_float(latest_signals.get('ema_50')),
                'above_ema': bool(latest_signals.get('above_ema', True)),
                'filtered': bool(latest_signals.get('trend_filtered', False)),
            },
            'thresholds': {
                'overweight': cfg.OVERWEIGHT_THRESHOLD,
                'underweight': cfg.UNDERWEIGHT_THRESHOLD,
            },
            'factors': factor_breakdown,
            'history': {
                'prices': price_history,
                'zscores': zscore_history,
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/factors")
async def get_factor_history():
    """
    Get historical factor z-scores for charting.

    Returns the last 180 days of factor values.
    """
    try:
        cfg = Config()

        # Fetch market data
        data = fetch_all_yahoo_data(cfg.TICKERS, days=450)

        if len(data) < cfg.ZSCORE_LOOKBACK:
            raise HTTPException(
                status_code=500,
                detail=f"Insufficient data. Need at least {cfg.ZSCORE_LOOKBACK} days."
            )

        # Fetch DVOL
        dvol = fetch_deribit_dvol()

        # Calculate factors
        factors = calculate_factors(data, dvol=dvol)

        # Get last 180 days
        chart_days = 180
        history = factors.tail(chart_days)

        factor_keys = ['equity_factor', 'volatility_factor', 'currency_factor',
                       'credit_factor', 'rates_factor']

        result = []
        for idx, row in history.iterrows():
            entry = {'date': idx.strftime('%Y-%m-%d')}
            for key in factor_keys:
                entry[key] = safe_float(row.get(key))
            result.append(entry)

        return JSONResponse(content={'factors': result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
