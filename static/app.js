/**
 * Bitcoin Factor Allocation Model - Frontend Application
 */

// Global chart instances
let priceChart = null;
let zscoreChart = null;

// Chart.js configuration
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            position: 'top',
            labels: {
                color: '#a0a0b0',
                font: { family: 'Inter', size: 11 },
                padding: 15,
                usePointStyle: true,
            }
        },
        tooltip: {
            backgroundColor: '#1a1a25',
            titleColor: '#ffffff',
            bodyColor: '#a0a0b0',
            borderColor: '#2a2a3a',
            borderWidth: 1,
            padding: 12,
            displayColors: true,
            callbacks: {}
        }
    },
    scales: {
        x: {
            type: 'time',
            time: {
                unit: 'day',
                displayFormats: { day: 'MMM d' }
            },
            grid: {
                color: 'rgba(42, 42, 58, 0.5)',
                drawBorder: false
            },
            ticks: {
                color: '#606070',
                font: { size: 10 },
                maxRotation: 45
            }
        },
        y: {
            grid: {
                color: 'rgba(42, 42, 58, 0.5)',
                drawBorder: false
            },
            ticks: {
                color: '#606070',
                font: { size: 10 }
            }
        }
    }
};

/**
 * Format currency values
 */
function formatCurrency(value) {
    if (value == null) return '--';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(value);
}

/**
 * Format numbers with sign
 */
function formatSignedNumber(value, decimals = 2) {
    if (value == null) return '--';
    const sign = value >= 0 ? '+' : '';
    return sign + value.toFixed(decimals);
}

/**
 * Update the signal display
 */
function updateSignalDisplay(data) {
    const signalCard = document.getElementById('signal-card');
    const signalValue = document.getElementById('signal-value');
    const zscoreValue = document.getElementById('zscore-value');
    const convictionValue = document.getElementById('conviction-value');
    const dataDate = document.getElementById('data-date');

    const signal = data.signal.current || 'NEUTRAL';

    // Update signal card styling
    signalCard.className = 'signal-card signal-' + signal.toLowerCase();

    // Update values
    signalValue.textContent = signal;
    zscoreValue.textContent = formatSignedNumber(data.signal.z_score);
    convictionValue.textContent = data.signal.conviction != null
        ? Math.round(data.signal.conviction) + '%'
        : '--%';
    dataDate.textContent = data.data_date || '--';
}

/**
 * Update trend filter display
 */
function updateTrendFilter(data) {
    const trendIcon = document.getElementById('trend-icon');
    const btcPrice = document.getElementById('btc-price');
    const emaPrice = document.getElementById('ema-price');
    const emaStatus = document.getElementById('ema-status');
    const filterNote = document.getElementById('filter-note');

    const trend = data.trend_filter;

    btcPrice.textContent = formatCurrency(trend.btc_price);
    emaPrice.textContent = formatCurrency(trend.ema_50);

    if (trend.above_ema) {
        trendIcon.textContent = '📈';
        emaStatus.textContent = 'ABOVE';
        emaStatus.className = 'price-value status-badge above';
    } else {
        trendIcon.textContent = '📉';
        emaStatus.textContent = 'BELOW';
        emaStatus.className = 'price-value status-badge below';
    }

    if (trend.filtered) {
        filterNote.textContent = '⚠ Signal filtered: Raw OVERWEIGHT capped to NEUTRAL (below EMA)';
    } else {
        filterNote.textContent = '';
    }
}

/**
 * Create/update the price chart
 */
function updatePriceChart(data) {
    const ctx = document.getElementById('price-chart').getContext('2d');

    const prices = data.history.prices;
    const labels = prices.map(p => new Date(p.date));
    const priceData = prices.map(p => p.price);
    const emaData = prices.map(p => p.ema50);

    // Create background colors based on signal
    const signalColors = {
        'OVERWEIGHT': 'rgba(34, 197, 94, 0.15)',
        'NEUTRAL': 'rgba(245, 158, 11, 0.15)',
        'UNDERWEIGHT': 'rgba(239, 68, 68, 0.15)'
    };

    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'BTC Price',
                    data: priceData,
                    borderColor: '#f7931a',
                    backgroundColor: 'rgba(247, 147, 26, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4
                },
                {
                    label: '50 EMA',
                    data: emaData,
                    borderColor: '#06b6d4',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 3
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + formatCurrency(context.raw);
                        }
                    }
                }
            },
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        callback: function(value) {
                            return '$' + (value / 1000).toFixed(0) + 'k';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create/update the z-score chart
 */
function updateZscoreChart(data) {
    const ctx = document.getElementById('zscore-chart').getContext('2d');

    const zscores = data.history.zscores;
    const labels = zscores.map(z => new Date(z.date));
    const zscoreData = zscores.map(z => z.zscore);

    // Create segment colors based on thresholds
    const segmentColors = zscoreData.map(z => {
        if (z > data.thresholds.overweight) return 'rgba(34, 197, 94, 0.8)';
        if (z < data.thresholds.underweight) return 'rgba(239, 68, 68, 0.8)';
        return 'rgba(245, 158, 11, 0.8)';
    });

    if (zscoreChart) {
        zscoreChart.destroy();
    }

    zscoreChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Composite Z-Score',
                    data: zscoreData,
                    backgroundColor: segmentColors,
                    borderColor: segmentColors.map(c => c.replace('0.8', '1')),
                    borderWidth: 1,
                    borderRadius: 2
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                annotation: {
                    annotations: {
                        overweightLine: {
                            type: 'line',
                            yMin: data.thresholds.overweight,
                            yMax: data.thresholds.overweight,
                            borderColor: 'rgba(34, 197, 94, 0.5)',
                            borderWidth: 1,
                            borderDash: [5, 5]
                        },
                        underweightLine: {
                            type: 'line',
                            yMin: data.thresholds.underweight,
                            yMax: data.thresholds.underweight,
                            borderColor: 'rgba(239, 68, 68, 0.5)',
                            borderWidth: 1,
                            borderDash: [5, 5]
                        }
                    }
                },
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            const z = context.raw;
                            let signal = 'NEUTRAL';
                            if (z > data.thresholds.overweight) signal = 'OVERWEIGHT';
                            if (z < data.thresholds.underweight) signal = 'UNDERWEIGHT';
                            return `Z-Score: ${formatSignedNumber(z)} (${signal})`;
                        }
                    }
                }
            },
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    min: -2.5,
                    max: 2.5,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        stepSize: 0.5,
                        callback: function(value) {
                            return formatSignedNumber(value, 1);
                        }
                    }
                }
            }
        }
    });
}

/**
 * Update factors display
 */
function updateFactors(data) {
    const grid = document.getElementById('factors-grid');
    grid.innerHTML = '';

    data.factors.forEach((factor, index) => {
        const card = document.createElement('div');
        card.className = 'factor-card fade-in';
        card.style.animationDelay = `${index * 0.1}s`;

        const signalClass = factor.signal.toLowerCase();
        const valueColor = factor.value != null
            ? (factor.value > 0 ? 'var(--signal-overweight)' : factor.value < 0 ? 'var(--signal-underweight)' : 'var(--text-primary)')
            : 'var(--text-muted)';

        // Build components HTML
        let componentsHtml = '';
        factor.components.forEach(comp => {
            const trendClass = comp.trend ? comp.trend.toLowerCase() : 'neutral';
            const trendSymbol = comp.trend === 'TRENDING' ? '↗' : comp.trend === 'REVERTING' ? '↺' : '→';

            componentsHtml += `
                <div class="component-row">
                    <span class="component-name">${comp.name}</span>
                    <span class="component-value">${formatSignedNumber(comp.z_score, 2)}</span>
                    <span class="component-value">H: ${comp.hurst != null ? comp.hurst.toFixed(2) : '--'}</span>
                    <span class="component-trend ${trendClass}">${trendSymbol} ${comp.trend || 'N/A'}</span>
                </div>
            `;
        });

        card.innerHTML = `
            <div class="factor-header">
                <div>
                    <div class="factor-name">${factor.name}</div>
                    <div class="factor-weight">Weight: ${factor.weight}%</div>
                </div>
                <span class="factor-signal ${signalClass}">${factor.signal}</span>
            </div>
            <div class="factor-value" style="color: ${valueColor}">
                ${factor.value != null ? formatSignedNumber(factor.value) : '--'}
            </div>
            <div class="components-list">
                ${componentsHtml}
            </div>
        `;

        grid.appendChild(card);
    });
}

/**
 * Update last updated time
 */
function updateLastUpdated() {
    const el = document.getElementById('last-updated-time');
    const now = new Date();
    el.textContent = now.toLocaleTimeString();
}

/**
 * Show error message
 */
function showError(message) {
    const banner = document.getElementById('error');
    const messageEl = document.getElementById('error-message');
    messageEl.textContent = message;
    banner.classList.remove('hidden');
}

/**
 * Hide error message
 */
function hideError() {
    const banner = document.getElementById('error');
    banner.classList.add('hidden');
}

/**
 * Show loading overlay
 */
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

/**
 * Fetch signal data from API
 */
async function fetchSignalData() {
    showLoading();
    hideError();

    try {
        const response = await fetch('/api/signal');

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to fetch signal data');
        }

        const data = await response.json();

        // Update all UI components
        updateSignalDisplay(data);
        updateTrendFilter(data);
        updatePriceChart(data);
        updateZscoreChart(data);
        updateFactors(data);
        updateLastUpdated();

        hideLoading();

    } catch (error) {
        console.error('Error fetching data:', error);
        hideLoading();
        showError(error.message || 'Failed to fetch data. Please try again.');
    }
}

/**
 * Initialize the application
 */
function init() {
    // Set up refresh button
    const refreshBtn = document.getElementById('refresh-btn');
    refreshBtn.addEventListener('click', fetchSignalData);

    // Set up error dismiss button
    const dismissBtn = document.getElementById('error-dismiss');
    dismissBtn.addEventListener('click', hideError);

    // Initial data fetch
    fetchSignalData();
}

// Start the app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
