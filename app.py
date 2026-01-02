"""
Bitcoin Factor Allocation Model - Web API
==========================================
FastAPI backend for serving the BTC factor model via REST API.
Includes caching for fast page loads.
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
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


# =============================================================================
# CACHE IMPLEMENTATION
# =============================================================================

class SignalCache:
    """
    In-memory cache for computed signal data.

    - Stores fully computed API response
    - Configurable TTL (time-to-live)
    - Thread-safe updates
    - Background refresh support
    """

    def __init__(self, ttl_minutes: int = 60):
        self.ttl = timedelta(minutes=ttl_minutes)
        self.data: Optional[Dict[str, Any]] = None
        self.last_updated: Optional[datetime] = None
        self.is_refreshing: bool = False
        self._lock = threading.Lock()

    def is_valid(self) -> bool:
        """Check if cache is valid (exists and not expired)."""
        if self.data is None or self.last_updated is None:
            return False
        return datetime.now() - self.last_updated < self.ttl

    def get(self) -> Optional[Dict[str, Any]]:
        """Get cached data if valid."""
        if self.is_valid():
            return self.data
        return None

    def set(self, data: Dict[str, Any]):
        """Update cache with new data."""
        with self._lock:
            self.data = data
            self.last_updated = datetime.now()
            self.is_refreshing = False

    def invalidate(self):
        """Invalidate the cache."""
        with self._lock:
            self.data = None
            self.last_updated = None

    def get_status(self) -> Dict[str, Any]:
        """Get cache status info."""
        return {
            'has_data': self.data is not None,
            'is_valid': self.is_valid(),
            'is_refreshing': self.is_refreshing,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'expires_at': (self.last_updated + self.ttl).isoformat() if self.last_updated else None,
            'ttl_minutes': self.ttl.total_seconds() / 60,
        }


# Global cache instance (1 hour TTL by default)
cache = SignalCache(ttl_minutes=60)


# =============================================================================
# FASTAPI APP
# =============================================================================

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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


def compute_signal_data() -> Dict[str, Any]:
    """
    Compute full signal data from market sources.
    This is the expensive operation that we cache.
    """
    cfg = Config()

    # Fetch market data
    data = fetch_all_yahoo_data(cfg.TICKERS, days=450)

    if len(data) < cfg.ZSCORE_LOOKBACK:
        raise ValueError(f"Insufficient data. Need at least {cfg.ZSCORE_LOOKBACK} days.")

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
    return {
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


def refresh_cache_sync():
    """Synchronously refresh the cache (for background tasks)."""
    try:
        cache.is_refreshing = True
        data = compute_signal_data()
        cache.set(data)
        print(f"[{datetime.now().isoformat()}] Cache refreshed successfully")
    except Exception as e:
        cache.is_refreshing = False
        print(f"[{datetime.now().isoformat()}] Cache refresh failed: {e}")
        raise


# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Pre-populate cache on startup."""
    print("Starting up - populating cache...")
    try:
        # Run in thread pool to not block startup
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, refresh_cache_sync)
        print("Cache populated successfully!")
    except Exception as e:
        print(f"Failed to populate cache on startup: {e}")
        print("Cache will be populated on first request.")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache": cache.get_status()
    }


@app.get("/api/signal")
async def get_signal(
    refresh: bool = Query(False, description="Force refresh the cache"),
    background_tasks: BackgroundTasks = None
):
    """
    Get current Bitcoin factor allocation signal.

    Returns the current signal (OVERWEIGHT/NEUTRAL/UNDERWEIGHT),
    composite z-score, factor breakdown, and component details.

    Query params:
    - refresh: Force refresh the cache (ignores TTL)
    """
    try:
        # Check if we need to refresh
        if refresh or not cache.is_valid():
            if cache.is_refreshing:
                # Another request is already refreshing
                # Return stale data if available, otherwise wait
                if cache.data:
                    response = cache.data.copy()
                    response['cache'] = {
                        'status': 'refreshing',
                        'stale': True,
                        **cache.get_status()
                    }
                    return JSONResponse(content=response)
                else:
                    # No cached data, must wait for refresh
                    refresh_cache_sync()
            else:
                # Perform refresh
                refresh_cache_sync()

        # Return cached data
        cached = cache.get()
        if cached:
            response = cached.copy()
            response['cache'] = {
                'status': 'hit',
                'stale': False,
                **cache.get_status()
            }
            return JSONResponse(content=response)

        # Shouldn't reach here, but fallback to compute
        data = compute_signal_data()
        cache.set(data)
        data['cache'] = {'status': 'miss', **cache.get_status()}
        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/refresh")
async def force_refresh(background_tasks: BackgroundTasks):
    """
    Force refresh the cache in the background.
    Returns immediately with cache status.
    """
    if cache.is_refreshing:
        return JSONResponse(content={
            'status': 'already_refreshing',
            'cache': cache.get_status()
        })

    # Start background refresh
    cache.is_refreshing = True
    background_tasks.add_task(refresh_cache_sync)

    return JSONResponse(content={
        'status': 'refresh_started',
        'cache': cache.get_status()
    })


@app.get("/api/cache/status")
async def get_cache_status():
    """Get current cache status."""
    return JSONResponse(content=cache.get_status())


@app.delete("/api/cache")
async def invalidate_cache():
    """Invalidate the cache (requires refresh on next request)."""
    cache.invalidate()
    return JSONResponse(content={
        'status': 'invalidated',
        'cache': cache.get_status()
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
