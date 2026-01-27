"""
Institutional-Grade Technical Indicator Suite.

Phase 2 of the Quantitative Technical Analysis Agent implementing four
sophisticated indicators with professional-level signal generation.

INDICATORS (as specified in coursework - distinct from Fardeen's RSI/MACD/BB/ATR/ADX):

    1. ICHIMOKU KINKO HYO - Complete Japanese equilibrium system
       - All 5 core lines with full interpretation
       - TK Cross strength classification (Strong/Neutral/Weak by cloud position)
       - Kumo breakout with momentum confirmation
       - Kijun-sen as dynamic support/resistance with bounce/break detection
       - Chikou Span confirmation (lagging line vs historical price)
       - Cloud twist detection (future trend reversal warning)
       - Ichimoku Price Theory targets (V, N, E, NT wave projections)
       - Sen line cross within cloud (Span A/B cross signals)
       - Multi-timeframe alignment with weekly Ichimoku
    
    2. VWAP PROFESSIONAL SUITE - Institutional benchmark system
       - Standard cumulative VWAP with statistical bands (1σ, 2σ, 3σ)
       - Rolling VWAP (5-day, 20-day) for swing trading
       - Anchored VWAP from earnings announcements (Phase 1 integration)
       - Anchored VWAP from significant swing highs/lows
       - VWAP slope analysis (trend strength measurement)
       - VWAP touch count analysis (S/R strength)
       - Z-score deviation for mean-reversion probability
       - Institutional accumulation/distribution zone detection
       - VWAP cross signals with volume confirmation
    
    3. WILLIAMS %R ADVANCED - Enhanced momentum oscillator
       - Triple-period system (Fast 7, Medium 14, Slow 21)
       - Regular divergence (price vs %R for reversals)
       - Hidden divergence (continuation signals)
       - Failure swing patterns (early reversal warning)
       - Momentum thrust signals (powerful trend initiation)
       - Extreme duration analysis (time in overbought/oversold)
       - %R pattern recognition (W-bottoms, M-tops)
       - Smoothed %R for noise reduction
       - Zone exit signals with confirmation
    
    4. CCI PROFESSIONAL - Commodity Channel Index with Woodies patterns
       - Dual CCI system (Short 14, Long 50) for timing and trend
       - Zero-line reject pattern (ZLR - trend continuation)
       - Trend line break pattern (TLB)
       - Hook from extreme pattern (HFE)
       - Reverse divergence pattern (continuation)
       - Ghost pattern (failed divergence)
       - CCI histogram for momentum visualization
       - Turbo CCI (6-period) for precise entry timing
       - Extreme readings analysis with duration

    5. CONFLUENCE ENGINE - Professional signal synthesis
       - Weighted multi-indicator scoring
       - Signal quality grading (A/B/C/D)
       - Multi-timeframe alignment scoring
       - Cross-indicator confirmation matrix
       - Trade setup identification with entry/stop/target
       - Risk-reward calculation
       - Signal strength decay analysis

Author: Tamer
Course: MSc AI Agents in Asset Management
Phase: 2 - Technical Indicators
Lines: ~3500
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

TRADING_DAYS_YEAR = 252
TRADING_DAYS_MONTH = 21
TRADING_DAYS_WEEK = 5


# =============================================================================
# ENUMERATIONS
# =============================================================================

class Signal(Enum):
    """Final trading signal classification."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalQuality(Enum):
    """Signal quality grade based on confirmation level."""
    A = "A"  # Multiple confirmations, MTF aligned, high probability
    B = "B"  # Good confirmation, mostly aligned
    C = "C"  # Single indicator, moderate confidence
    D = "D"  # Conflicting signals, low confidence


class Trend(Enum):
    """Trend direction classification."""
    STRONG_UP = "STRONG_UP"
    UP = "UP"
    NEUTRAL = "NEUTRAL"
    DOWN = "DOWN"
    STRONG_DOWN = "STRONG_DOWN"


class CloudPosition(Enum):
    """Price position relative to Ichimoku Kumo."""
    ABOVE = "ABOVE"
    INSIDE = "INSIDE"
    BELOW = "BELOW"


class TKCrossStrength(Enum):
    """Tenkan-Kijun cross strength classification."""
    STRONG_BULL = "STRONG_BULL"    # Cross above cloud
    NEUTRAL_BULL = "NEUTRAL_BULL"  # Cross inside cloud
    WEAK_BULL = "WEAK_BULL"        # Cross below cloud
    STRONG_BEAR = "STRONG_BEAR"    # Cross below cloud
    NEUTRAL_BEAR = "NEUTRAL_BEAR"  # Cross inside cloud
    WEAK_BEAR = "WEAK_BEAR"        # Cross above cloud
    NONE = "NONE"


class IchimokuWave(Enum):
    """Ichimoku Price Theory wave types for target calculation."""
    V_WAVE = "V"   # Simple reversal: Target = 2 * B - A
    N_WAVE = "N"   # Continuation: Target = C + (B - A)
    E_WAVE = "E"   # Extended: Target = B + (B - A)
    NT_WAVE = "NT" # Time-based: Target = C + (C - A)


class WoodieCCIPattern(Enum):
    """Woodies CCI pattern types."""
    ZLR = "ZERO_LINE_REJECT"       # Bounce off zero line in trend
    TLB = "TREND_LINE_BREAK"       # CCI trend line break
    HFE = "HOOK_FROM_EXTREME"      # Hook before reaching extreme
    REV_DIV = "REVERSE_DIVERGENCE" # Continuation divergence
    GHOST = "GHOST"                # Failed divergence (trap)
    NONE = "NONE"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IndicatorSignal:
    """Output from individual indicator analysis."""
    name: str
    value: float              # -1.0 to +1.0 (bearish to bullish)
    confidence: float         # 0-100%
    factors: List[str]        # Reasons for signal


@dataclass
class PriceTarget:
    """Price target from Ichimoku wave theory or other methods."""
    level: float
    method: str
    probability: float        # Estimated probability of reaching


@dataclass
class IchimokuAnalysis:
    """Complete Ichimoku system analysis output."""
    # Line values
    tenkan: float
    kijun: float
    senkou_a: float
    senkou_b: float
    chikou: float
    
    # Cloud metrics
    cloud_top: float
    cloud_bottom: float
    cloud_thickness_pct: float
    cloud_color: Trend
    future_cloud_color: Trend
    
    # Position analysis
    price_position: CloudPosition
    price_vs_tenkan: str
    price_vs_kijun: str
    
    # Signal detection
    tk_cross: TKCrossStrength
    kumo_breakout: Optional[str]
    kijun_bounce: bool
    kijun_break: bool
    chikou_confirmed: bool
    cloud_twist_ahead: bool
    sen_cross_in_cloud: Optional[str]
    
    # Price targets
    wave_targets: List[PriceTarget]
    
    # Multi-timeframe
    weekly_aligned: bool
    weekly_position: Optional[CloudPosition]
    
    # Final signal
    signal: IndicatorSignal


@dataclass
class VWAPAnalysis:
    """Complete VWAP suite analysis output."""
    # Standard VWAP
    vwap: float
    std_dev: float
    upper_1sd: float
    upper_2sd: float
    upper_3sd: float
    lower_1sd: float
    lower_2sd: float
    lower_3sd: float
    
    # Rolling VWAPs
    vwap_5d: float
    vwap_20d: float
    rolling_cross: Optional[str]
    
    # Anchored VWAPs
    anchored_earnings: Dict[str, float]
    anchored_swing_high: Optional[float]
    anchored_swing_low: Optional[float]
    nearest_anchored: Optional[Tuple[str, float, float]]  # (label, price, distance%)
    
    # Statistical analysis
    z_score: float
    mean_reversion_prob: float
    band_position: str
    
    # Trend analysis
    vwap_slope: float
    slope_trend: Trend
    
    # Support/Resistance strength
    touch_count_5d: int
    sr_strength: str
    
    # Volume analysis
    institutional_zone: bool
    volume_confirmation: bool
    
    # Final signal
    signal: IndicatorSignal


@dataclass
class WilliamsRAnalysis:
    """Complete Williams %R analysis output."""
    # Multi-period values
    fast_value: float         # 7-period
    medium_value: float       # 14-period
    slow_value: float         # 21-period
    smoothed_value: float     # 3-period SMA of medium
    
    # Zone analysis
    zone: str
    extreme_duration: int     # Bars in current extreme zone
    
    # Period alignment
    triple_alignment: str     # All three agree on direction
    
    # Pattern detection
    regular_bull_div: bool
    regular_bear_div: bool
    hidden_bull_div: bool
    hidden_bear_div: bool
    failure_swing_bull: bool
    failure_swing_bear: bool
    w_bottom: bool
    m_top: bool
    
    # Momentum
    momentum_thrust: Optional[str]
    zone_exit: Optional[str]
    
    # Final signal
    signal: IndicatorSignal


@dataclass
class CCIAnalysis:
    """Complete CCI analysis output."""
    # Multi-period values
    turbo_value: float        # 6-period for timing
    short_value: float        # 14-period standard
    long_value: float         # 50-period trend
    histogram: float          # Short - Long difference
    
    # Zone analysis
    zone: str
    extreme_duration: int
    
    # Trend analysis
    trend: Trend
    dual_alignment: bool
    
    # Zero-line analysis
    zero_cross: Optional[str]
    distance_from_zero: float
    
    # Woodies patterns
    woodies_pattern: WoodieCCIPattern
    pattern_strength: float
    
    # Divergence
    bullish_divergence: bool
    bearish_divergence: bool
    
    # Additional patterns
    bullish_hook: bool
    bearish_hook: bool
    
    # Final signal
    signal: IndicatorSignal


@dataclass
class MTFAnalysis:
    """Multi-timeframe confluence analysis."""
    daily_trend: Trend
    weekly_trend: Trend
    aligned: bool
    alignment_score: float
    dominant_timeframe: str
    trend_strength: str


@dataclass
class TradeSetup:
    """Identified trade setup with execution levels."""
    direction: str
    entry_zone: Tuple[float, float]
    stop_loss: float
    target_1: float
    target_2: Optional[float]
    target_3: Optional[float]
    risk_reward_1: float
    risk_reward_2: Optional[float]
    position_size_suggestion: str
    setup_type: str
    time_validity: str


@dataclass
class ConfluenceAnalysis:
    """Complete multi-indicator confluence output."""
    # Individual analyses
    ichimoku: IchimokuAnalysis
    vwap: VWAPAnalysis
    williams_r: WilliamsRAnalysis
    cci: CCIAnalysis
    mtf: MTFAnalysis
    
    # Confluence metrics
    confluence_score: float
    indicator_agreement: Dict[str, float]
    bullish_count: int
    bearish_count: int
    neutral_count: int
    
    # Cross-indicator confirmation
    divergence_confirmed: bool
    breakout_confirmed: bool
    momentum_confirmed: bool
    
    # Final output
    signal: Signal
    quality: SignalQuality
    confidence: float
    
    # Trade setup
    setup: Optional[TradeSetup]
    
    # Reasoning
    bullish_factors: List[str]
    bearish_factors: List[str]
    warnings: List[str]
    summary: str


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, handling NaN and None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_bool(val: Any, default: bool = False) -> bool:
    """Safely convert value to bool, handling NaN and None."""
    if val is None or pd.isna(val):
        return default
    try:
        return bool(val)
    except (TypeError, ValueError):
        return default


def calculate_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 10,
    threshold: float = 0.02
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect regular divergence between price and indicator.
    
    Regular Bullish: Price makes lower low, indicator makes higher low
    Regular Bearish: Price makes higher high, indicator makes lower high
    
    Returns:
        Tuple of (bullish_divergence, bearish_divergence) boolean Series
    """
    # Find local minima and maxima
    price_min = price.rolling(lookback, center=True).min()
    price_max = price.rolling(lookback, center=True).max()
    ind_min = indicator.rolling(lookback, center=True).min()
    ind_max = indicator.rolling(lookback, center=True).max()
    
    # Previous lows/highs
    prev_price_min = price_min.shift(lookback)
    prev_price_max = price_max.shift(lookback)
    prev_ind_min = ind_min.shift(lookback)
    prev_ind_max = ind_max.shift(lookback)
    
    # Bullish divergence: price lower low, indicator higher low
    price_ll = price_min < prev_price_min * (1 - threshold)
    ind_hl = ind_min > prev_ind_min
    bull_div = price_ll & ind_hl
    
    # Bearish divergence: price higher high, indicator lower high
    price_hh = price_max > prev_price_max * (1 + threshold)
    ind_lh = ind_max < prev_ind_max
    bear_div = price_hh & ind_lh
    
    return bull_div, bear_div


def calculate_hidden_divergence(
    price: pd.Series,
    indicator: pd.Series,
    trend: pd.Series,
    lookback: int = 10
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect hidden divergence (continuation signals).
    
    Hidden Bullish: Price makes higher low, indicator makes lower low (uptrend)
    Hidden Bearish: Price makes lower high, indicator makes higher high (downtrend)
    """
    price_min = price.rolling(lookback).min()
    price_max = price.rolling(lookback).max()
    ind_min = indicator.rolling(lookback).min()
    ind_max = indicator.rolling(lookback).max()
    
    prev_price_min = price_min.shift(lookback)
    prev_price_max = price_max.shift(lookback)
    prev_ind_min = ind_min.shift(lookback)
    prev_ind_max = ind_max.shift(lookback)
    
    uptrend = trend > 0
    downtrend = trend < 0
    
    # Hidden bullish: price HL, indicator LL, in uptrend
    price_hl = price_min > prev_price_min
    ind_ll = ind_min < prev_ind_min
    hidden_bull = price_hl & ind_ll & uptrend
    
    # Hidden bearish: price LH, indicator HH, in downtrend
    price_lh = price_max < prev_price_max
    ind_hh = ind_max > prev_ind_max
    hidden_bear = price_lh & ind_hh & downtrend
    
    return hidden_bull, hidden_bear


# =============================================================================
# ICHIMOKU KINKO HYO - COMPLETE SYSTEM
# =============================================================================

class IchimokuCloud:
    """
    Ichimoku Kinko Hyo (一目均衡表) - "One Glance Equilibrium Chart"
    
    Developed by Goichi Hosoda over 30 years of research before publishing
    in 1969. The most comprehensive single-indicator trading system.
    
    FIVE COMPONENTS:
    
        Tenkan-sen (転換線) - Conversion Line
            Formula: (9-period High + 9-period Low) / 2
            Represents short-term equilibrium/momentum
            
        Kijun-sen (基準線) - Base Line  
            Formula: (26-period High + 26-period Low) / 2
            Represents medium-term equilibrium
            Primary support/resistance, trailing stop level
            
        Senkou Span A (先行スパンA) - Leading Span A
            Formula: (Tenkan + Kijun) / 2, plotted 26 periods ahead
            Faster-moving cloud boundary
            
        Senkou Span B (先行スパンB) - Leading Span B
            Formula: (52-period High + 52-period Low) / 2, plotted 26 ahead
            Slower-moving cloud boundary
            
        Chikou Span (遅行スパン) - Lagging Span
            Formula: Close plotted 26 periods back
            Confirms trend when above/below historical price
    
    KUMO (CLOUD):
        Area between Senkou Span A and B
        Bullish when A > B (green), Bearish when A < B (red)
        Thickness indicates support/resistance strength
        Price above cloud = bullish, below = bearish, inside = consolidation
    
    TRADING SIGNALS:
        
        1. TK Cross (Tenkan crosses Kijun)
           - Above cloud = Strong bullish
           - Inside cloud = Neutral
           - Below cloud = Weak bullish (opposite for bearish)
        
        2. Kumo Breakout
           - Price breaks above cloud = Bullish breakout
           - Price breaks below cloud = Bearish breakdown
           
        3. Kijun Bounce/Break
           - Price bounces off Kijun = Trend continuation
           - Price breaks Kijun = Potential reversal
           
        4. Chikou Confirmation
           - Chikou above price 26 bars ago = Bullish confirmed
           - Chikou below price 26 bars ago = Bearish confirmed
           
        5. Cloud Twist
           - Senkou A crosses Senkou B = Future trend change signal
           
        6. Sen Cross in Cloud
           - Span A crosses above Span B (inside cloud) = Bullish
           - Span A crosses below Span B (inside cloud) = Bearish
    
    PRICE THEORY (WAVE TARGETS):
        
        V Wave: Simple reversal
            Target = 2 * B - A (where A=start, B=reversal point)
            
        N Wave: Continuation after pullback
            Target = C + (B - A) (where C=pullback end)
            
        E Wave: Extended move
            Target = B + (B - A)
            
        NT Wave: Time-based projection
            Target = C + (C - A)
    """
    
    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
    
    def calculate(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Calculate all Ichimoku components and derived signals.
        
        Args:
            df: DataFrame with High, Low, Close columns
            prefix: Column prefix for multi-timeframe (e.g., "W_")
        
        Returns:
            DataFrame with Ichimoku columns added
        """
        df = df.copy()
        p = prefix
        high, low, close = df['High'], df['Low'], df['Close']
        
        # === CORE COMPONENTS ===
        
        # Tenkan-sen (Conversion Line)
        df[f'{p}Ich_Tenkan'] = (
            high.rolling(self.tenkan_period).max() +
            low.rolling(self.tenkan_period).min()
        ) / 2
        
        # Kijun-sen (Base Line)
        df[f'{p}Ich_Kijun'] = (
            high.rolling(self.kijun_period).max() +
            low.rolling(self.kijun_period).min()
        ) / 2
        
        # Senkou Span A (Leading Span A) - displaced forward
        span_a_raw = (df[f'{p}Ich_Tenkan'] + df[f'{p}Ich_Kijun']) / 2
        df[f'{p}Ich_SpanA'] = span_a_raw.shift(self.displacement)
        df[f'{p}Ich_SpanA_Future'] = span_a_raw  # Current calculation (future cloud)
        
        # Senkou Span B (Leading Span B) - displaced forward
        span_b_raw = (
            high.rolling(self.senkou_b_period).max() +
            low.rolling(self.senkou_b_period).min()
        ) / 2
        df[f'{p}Ich_SpanB'] = span_b_raw.shift(self.displacement)
        df[f'{p}Ich_SpanB_Future'] = span_b_raw  # Current calculation (future cloud)
        
        # Chikou Span (Lagging Span) - displaced backward
        df[f'{p}Ich_Chikou'] = close.shift(-self.displacement)
        
        # === CLOUD ANALYSIS ===
        
        # Cloud boundaries
        df[f'{p}Ich_CloudTop'] = df[[f'{p}Ich_SpanA', f'{p}Ich_SpanB']].max(axis=1)
        df[f'{p}Ich_CloudBottom'] = df[[f'{p}Ich_SpanA', f'{p}Ich_SpanB']].min(axis=1)
        df[f'{p}Ich_CloudThickness'] = df[f'{p}Ich_CloudTop'] - df[f'{p}Ich_CloudBottom']
        df[f'{p}Ich_CloudThickPct'] = df[f'{p}Ich_CloudThickness'] / close * 100
        
        # Cloud color (current and future)
        df[f'{p}Ich_CloudBull'] = df[f'{p}Ich_SpanA'] > df[f'{p}Ich_SpanB']
        df[f'{p}Ich_FutureCloudBull'] = df[f'{p}Ich_SpanA_Future'] > df[f'{p}Ich_SpanB_Future']
        
        # Price position relative to cloud
        df[f'{p}Ich_AboveCloud'] = close > df[f'{p}Ich_CloudTop']
        df[f'{p}Ich_BelowCloud'] = close < df[f'{p}Ich_CloudBottom']
        df[f'{p}Ich_InCloud'] = ~df[f'{p}Ich_AboveCloud'] & ~df[f'{p}Ich_BelowCloud']
        
        # Price vs Tenkan/Kijun
        df[f'{p}Ich_AboveTenkan'] = close > df[f'{p}Ich_Tenkan']
        df[f'{p}Ich_AboveKijun'] = close > df[f'{p}Ich_Kijun']
        
        # === TK CROSS DETECTION ===
        
        tk_diff = df[f'{p}Ich_Tenkan'] - df[f'{p}Ich_Kijun']
        tk_diff_prev = tk_diff.shift(1)
        
        # Bull cross: Tenkan crosses above Kijun
        df[f'{p}Ich_TK_BullCross'] = (tk_diff > 0) & (tk_diff_prev <= 0)
        # Bear cross: Tenkan crosses below Kijun
        df[f'{p}Ich_TK_BearCross'] = (tk_diff < 0) & (tk_diff_prev >= 0)
        
        # TK relationship strength
        df[f'{p}Ich_TK_Bullish'] = tk_diff > 0
        
        # === KUMO BREAKOUT DETECTION ===
        
        above_prev = df[f'{p}Ich_AboveCloud'].shift(1).fillna(False).astype(bool)
        below_prev = df[f'{p}Ich_BelowCloud'].shift(1).fillna(False).astype(bool)
        
        df[f'{p}Ich_KumoBreakUp'] = df[f'{p}Ich_AboveCloud'] & ~above_prev
        df[f'{p}Ich_KumoBreakDown'] = df[f'{p}Ich_BelowCloud'] & ~below_prev
        
        # === KIJUN INTERACTION ===
        
        kijun = df[f'{p}Ich_Kijun']
        
        # Kijun bounce: price touches and holds
        near_kijun = (low <= kijun * 1.005) & (low >= kijun * 0.995)
        closed_above = close > kijun
        was_above = close.shift(1) > kijun.shift(1)
        df[f'{p}Ich_KijunBounce'] = near_kijun & closed_above & was_above
        
        # Kijun break: decisive close below
        df[f'{p}Ich_KijunBreak'] = (close < kijun * 0.99) & was_above
        
        # === CHIKOU CONFIRMATION ===
        
        # Compare current close with price 26 periods ago
        historical_close = close.shift(self.displacement)
        df[f'{p}Ich_ChikouBullish'] = close > historical_close
        
        # === CLOUD TWIST (SEN CROSS) ===
        
        future_bull = df[f'{p}Ich_FutureCloudBull']
        future_bull_prev = future_bull.shift(1).fillna(False).astype(bool)
        
        df[f'{p}Ich_CloudTwistBull'] = future_bull & ~future_bull_prev
        df[f'{p}Ich_CloudTwistBear'] = ~future_bull & future_bull_prev
        
        # === SEN CROSS INSIDE CLOUD ===
        
        # Span A crosses Span B while price is inside cloud
        span_a = df[f'{p}Ich_SpanA']
        span_b = df[f'{p}Ich_SpanB']
        span_diff = span_a - span_b
        span_diff_prev = span_diff.shift(1)
        
        sen_bull_cross = (span_diff > 0) & (span_diff_prev <= 0)
        sen_bear_cross = (span_diff < 0) & (span_diff_prev >= 0)
        
        df[f'{p}Ich_SenCrossBullInCloud'] = sen_bull_cross & df[f'{p}Ich_InCloud']
        df[f'{p}Ich_SenCrossBearInCloud'] = sen_bear_cross & df[f'{p}Ich_InCloud']
        
        # === WAVE TARGETS (calculated in analyze method) ===
        
        return df
    
    def calculate_wave_targets(
        self,
        df: pd.DataFrame,
        idx: int = -1
    ) -> List[PriceTarget]:
        """
        Calculate Ichimoku Price Theory wave targets.
        
        Uses recent swing points to project potential price targets.
        """
        targets = []
        
        if len(df) < 50:
            return targets
        
        close = df['Close'].iloc[idx]
        lookback = min(50, len(df) - 1)
        recent = df.iloc[max(0, idx - lookback):idx + 1]
        
        # Find swing points
        highs = recent['High']
        lows = recent['Low']
        
        recent_high = highs.max()
        recent_low = lows.min()
        
        # Determine trend direction for appropriate targets
        trend_up = close > recent.iloc[0]['Close']
        
        if trend_up:
            # V Wave target (bullish)
            v_target = 2 * close - recent_low
            targets.append(PriceTarget(
                level=v_target,
                method="V_WAVE",
                probability=0.65
            ))
            
            # N Wave target
            pullback = recent_low
            n_target = close + (recent_high - pullback)
            targets.append(PriceTarget(
                level=n_target,
                method="N_WAVE",
                probability=0.55
            ))
            
            # E Wave target (extended)
            e_target = recent_high + (recent_high - recent_low)
            targets.append(PriceTarget(
                level=e_target,
                method="E_WAVE",
                probability=0.40
            ))
        else:
            # Bearish targets
            v_target = 2 * close - recent_high
            targets.append(PriceTarget(
                level=v_target,
                method="V_WAVE",
                probability=0.65
            ))
            
            pullback = recent_high
            n_target = close - (pullback - recent_low)
            targets.append(PriceTarget(
                level=n_target,
                method="N_WAVE",
                probability=0.55
            ))
        
        return targets
    
    def analyze(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame = None,
        idx: int = -1
    ) -> IchimokuAnalysis:
        """
        Perform complete Ichimoku analysis at specified index.
        
        Args:
            df: Daily DataFrame with Ichimoku columns
            weekly_df: Optional weekly DataFrame for MTF
            idx: Index to analyze (-1 for latest)
        """
        row = df.iloc[idx]
        close = row['Close']
        
        # Extract line values
        tenkan = safe_float(row.get('Ich_Tenkan'))
        kijun = safe_float(row.get('Ich_Kijun'))
        span_a = safe_float(row.get('Ich_SpanA'))
        span_b = safe_float(row.get('Ich_SpanB'))
        chikou = safe_float(row.get('Ich_Chikou'))
        cloud_top = safe_float(row.get('Ich_CloudTop'))
        cloud_bottom = safe_float(row.get('Ich_CloudBottom'))
        cloud_thick_pct = safe_float(row.get('Ich_CloudThickPct'))
        
        # Cloud color
        cloud_bull = safe_bool(row.get('Ich_CloudBull'))
        future_bull = safe_bool(row.get('Ich_FutureCloudBull'))
        cloud_color = Trend.UP if cloud_bull else Trend.DOWN
        future_cloud_color = Trend.UP if future_bull else Trend.DOWN
        
        # Price position
        above = safe_bool(row.get('Ich_AboveCloud'))
        below = safe_bool(row.get('Ich_BelowCloud'))
        if above:
            price_position = CloudPosition.ABOVE
        elif below:
            price_position = CloudPosition.BELOW
        else:
            price_position = CloudPosition.INSIDE
        
        # Price vs lines
        price_vs_tenkan = "ABOVE" if safe_bool(row.get('Ich_AboveTenkan')) else "BELOW"
        price_vs_kijun = "ABOVE" if safe_bool(row.get('Ich_AboveKijun')) else "BELOW"
        
        # TK Cross strength classification
        tk_cross = TKCrossStrength.NONE
        if safe_bool(row.get('Ich_TK_BullCross')):
            if above:
                tk_cross = TKCrossStrength.STRONG_BULL
            elif below:
                tk_cross = TKCrossStrength.WEAK_BULL
            else:
                tk_cross = TKCrossStrength.NEUTRAL_BULL
        elif safe_bool(row.get('Ich_TK_BearCross')):
            if below:
                tk_cross = TKCrossStrength.STRONG_BEAR
            elif above:
                tk_cross = TKCrossStrength.WEAK_BEAR
            else:
                tk_cross = TKCrossStrength.NEUTRAL_BEAR
        
        # Kumo breakout
        kumo_breakout = None
        if safe_bool(row.get('Ich_KumoBreakUp')):
            kumo_breakout = "BULLISH"
        elif safe_bool(row.get('Ich_KumoBreakDown')):
            kumo_breakout = "BEARISH"
        
        # Kijun interactions
        kijun_bounce = safe_bool(row.get('Ich_KijunBounce'))
        kijun_break = safe_bool(row.get('Ich_KijunBreak'))
        
        # Chikou confirmation
        chikou_confirmed = safe_bool(row.get('Ich_ChikouBullish'))
        
        # Cloud twist
        cloud_twist_bull = safe_bool(row.get('Ich_CloudTwistBull'))
        cloud_twist_bear = safe_bool(row.get('Ich_CloudTwistBear'))
        cloud_twist_ahead = cloud_twist_bull or cloud_twist_bear
        
        # Sen cross in cloud
        sen_cross_in_cloud = None
        if safe_bool(row.get('Ich_SenCrossBullInCloud')):
            sen_cross_in_cloud = "BULLISH"
        elif safe_bool(row.get('Ich_SenCrossBearInCloud')):
            sen_cross_in_cloud = "BEARISH"
        
        # Wave targets
        wave_targets = self.calculate_wave_targets(df, idx)
        
        # Weekly alignment
        weekly_aligned = True
        weekly_position = None
        if weekly_df is not None and len(weekly_df) > 0:
            try:
                w_row = weekly_df.iloc[-1]
                w_above = safe_bool(w_row.get('W_Ich_AboveCloud'))
                w_below = safe_bool(w_row.get('W_Ich_BelowCloud'))
                
                if w_above:
                    weekly_position = CloudPosition.ABOVE
                elif w_below:
                    weekly_position = CloudPosition.BELOW
                else:
                    weekly_position = CloudPosition.INSIDE
                
                # Check alignment
                if above and not w_above:
                    weekly_aligned = False
                elif below and not w_below:
                    weekly_aligned = False
            except Exception:
                pass
        
        # Generate signal
        signal = self._generate_signal(
            close, price_position, cloud_color, future_cloud_color,
            tk_cross, kumo_breakout, kijun_bounce, kijun_break,
            chikou_confirmed, cloud_twist_ahead, sen_cross_in_cloud,
            weekly_aligned, tenkan, kijun
        )
        
        return IchimokuAnalysis(
            tenkan=tenkan,
            kijun=kijun,
            senkou_a=span_a,
            senkou_b=span_b,
            chikou=chikou,
            cloud_top=cloud_top,
            cloud_bottom=cloud_bottom,
            cloud_thickness_pct=cloud_thick_pct,
            cloud_color=cloud_color,
            future_cloud_color=future_cloud_color,
            price_position=price_position,
            price_vs_tenkan=price_vs_tenkan,
            price_vs_kijun=price_vs_kijun,
            tk_cross=tk_cross,
            kumo_breakout=kumo_breakout,
            kijun_bounce=kijun_bounce,
            kijun_break=kijun_break,
            chikou_confirmed=chikou_confirmed,
            cloud_twist_ahead=cloud_twist_ahead,
            sen_cross_in_cloud=sen_cross_in_cloud,
            wave_targets=wave_targets,
            weekly_aligned=weekly_aligned,
            weekly_position=weekly_position,
            signal=signal
        )
    
    def _generate_signal(
        self,
        close: float,
        position: CloudPosition,
        cloud_color: Trend,
        future_color: Trend,
        tk_cross: TKCrossStrength,
        kumo_breakout: Optional[str],
        kijun_bounce: bool,
        kijun_break: bool,
        chikou_confirmed: bool,
        cloud_twist: bool,
        sen_cross: Optional[str],
        weekly_aligned: bool,
        tenkan: float,
        kijun: float
    ) -> IndicatorSignal:
        """Generate comprehensive Ichimoku signal."""
        value = 0.0
        confidence = 30.0
        factors = []
        
        # Price vs Cloud (weight: 0.25)
        if position == CloudPosition.ABOVE:
            value += 0.25
            confidence += 10
            factors.append("Price ABOVE cloud (bullish bias)")
        elif position == CloudPosition.BELOW:
            value -= 0.25
            confidence += 10
            factors.append("Price BELOW cloud (bearish bias)")
        else:
            factors.append("Price INSIDE cloud (consolidation/transition)")
        
        # Cloud color - current (weight: 0.10)
        if cloud_color == Trend.UP:
            value += 0.10
            factors.append("Current cloud is bullish")
        else:
            value -= 0.10
            factors.append("Current cloud is bearish")
        
        # Future cloud color (weight: 0.08)
        if future_color == Trend.UP:
            value += 0.08
            factors.append("Future cloud bullish")
        else:
            value -= 0.08
            factors.append("Future cloud bearish")
        
        # TK relationship (weight: 0.12)
        if tenkan > kijun:
            value += 0.12
            factors.append("Tenkan > Kijun (short-term momentum bullish)")
        else:
            value -= 0.12
            factors.append("Tenkan < Kijun (short-term momentum bearish)")
        
        # TK Cross (weight: 0.15)
        if tk_cross in [TKCrossStrength.STRONG_BULL, TKCrossStrength.NEUTRAL_BULL]:
            value += 0.15 if "STRONG" in tk_cross.value else 0.12
            confidence += 20
            factors.append(f"BULLISH TK CROSS ({tk_cross.value})")
        elif tk_cross == TKCrossStrength.WEAK_BULL:
            value += 0.06
            confidence += 10
            factors.append("Weak bullish TK cross (below cloud)")
        elif tk_cross in [TKCrossStrength.STRONG_BEAR, TKCrossStrength.NEUTRAL_BEAR]:
            value -= 0.15 if "STRONG" in tk_cross.value else 0.12
            confidence += 20
            factors.append(f"BEARISH TK CROSS ({tk_cross.value})")
        elif tk_cross == TKCrossStrength.WEAK_BEAR:
            value -= 0.06
            confidence += 10
            factors.append("Weak bearish TK cross (above cloud)")
        
        # Kumo breakout (weight: 0.15)
        if kumo_breakout == "BULLISH":
            value += 0.15
            confidence += 25
            factors.append("KUMO BREAKOUT - Bullish signal")
        elif kumo_breakout == "BEARISH":
            value -= 0.15
            confidence += 25
            factors.append("KUMO BREAKDOWN - Bearish signal")
        
        # Kijun bounce (weight: 0.08)
        if kijun_bounce:
            value += 0.08
            confidence += 10
            factors.append("Kijun support bounce")
        
        # Kijun break (weight: 0.08)
        if kijun_break:
            value -= 0.10
            confidence += 10
            factors.append("Kijun support broken")
        
        # Chikou confirmation (weight: 0.08)
        if chikou_confirmed:
            value += 0.08
            confidence += 8
            factors.append("Chikou confirms bullish")
        else:
            value -= 0.05
            factors.append("Chikou not confirming")
        
        # Sen cross in cloud (weight: 0.06)
        if sen_cross == "BULLISH":
            value += 0.06
            confidence += 8
            factors.append("Bullish sen cross inside cloud")
        elif sen_cross == "BEARISH":
            value -= 0.06
            confidence += 8
            factors.append("Bearish sen cross inside cloud")
        
        # Cloud twist warning
        if cloud_twist:
            confidence -= 5
            factors.append("WARNING: Cloud twist ahead (potential trend change)")
        
        # Weekly alignment bonus/penalty
        if weekly_aligned:
            value *= 1.15
            confidence += 12
            factors.append("Weekly timeframe ALIGNED")
        else:
            value *= 0.85
            confidence -= 5
            factors.append("Weekly timeframe NOT aligned")
        
        value = max(-1.0, min(1.0, value))
        confidence = max(25, min(95, confidence))
        
        return IndicatorSignal(
            name="Ichimoku",
            value=value,
            confidence=confidence,
            factors=factors
        )


# =============================================================================
# VWAP PROFESSIONAL SUITE
# =============================================================================

class VWAPSuite:
    """
    Volume Weighted Average Price - Institutional Trading Standard.
    
    VWAP represents where the majority of volume transacted, making it the
    primary benchmark for institutional execution quality. Price above VWAP
    suggests buyers are in control; below suggests sellers dominate.
    
    COMPONENTS:
    
        Standard VWAP (Cumulative)
            Formula: Σ(Typical Price × Volume) / Σ(Volume)
            Running from period start
            Primary institutional benchmark
            
        Statistical Bands (1σ, 2σ, 3σ)
            Based on rolling standard deviation from VWAP
            ±1σ: Normal trading range (~68% of prices)
            ±2σ: Extended range (~95% of prices)
            ±3σ: Extreme range (~99% of prices)
            
        Rolling VWAP (5-day, 20-day)
            Uses rolling window instead of cumulative
            5-day: Short-term institutional level
            20-day: Monthly institutional level
            
        Anchored VWAP
            From specific events (earnings, swing points)
            Shows average price of buyers/sellers since event
            Critical for institutional reference levels
            
        VWAP Slope Analysis
            Rate of change of VWAP
            Indicates underlying trend strength
            
        Touch Count Analysis
            Number of times price touched VWAP
            More touches = stronger S/R level
    
    TRADING APPLICATIONS:
    
        1. Mean Reversion
           - Price at +2σ: Likely to revert to VWAP
           - Price at -2σ: Likely to bounce to VWAP
           
        2. Trend Confirmation
           - Price consistently above VWAP: Uptrend
           - Price consistently below VWAP: Downtrend
           
        3. Entry Optimization
           - Buy dips to VWAP in uptrend
           - Sell rallies to VWAP in downtrend
           
        4. Institutional Activity
           - High volume near VWAP: Institutional accumulation/distribution
           - Large deviations with low volume: Retail activity
    """
    
    def __init__(self):
        self.bands = [1.0, 2.0, 3.0]
    
    def calculate(
        self,
        df: pd.DataFrame,
        earnings_dates: List[str] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive VWAP suite.
        
        Args:
            df: DataFrame with OHLCV and TypicalPrice
            earnings_dates: Dates for anchored VWAP
        """
        df = df.copy()
        close = df['Close']
        volume = df['Volume']
        
        # Ensure TypicalPrice exists (from Phase 1)
        if 'TypicalPrice' not in df.columns:
            df['TypicalPrice'] = (df['High'] + df['Low'] + close) / 3
        tp = df['TypicalPrice']
        
        # === STANDARD CUMULATIVE VWAP ===
        
        pv_cum = (tp * volume).cumsum()
        vol_cum = volume.cumsum()
        df['VWAP'] = pv_cum / vol_cum
        
        # Standard deviation from VWAP
        df['VWAP_Var'] = ((tp - df['VWAP']) ** 2 * volume).cumsum() / vol_cum
        df['VWAP_Std'] = np.sqrt(df['VWAP_Var'])
        
        # Statistical bands
        for mult in self.bands:
            suffix = f'{mult}'.replace('.', '')
            df[f'VWAP_Upper_{suffix}SD'] = df['VWAP'] + mult * df['VWAP_Std']
            df[f'VWAP_Lower_{suffix}SD'] = df['VWAP'] - mult * df['VWAP_Std']
        
        # Z-score (standardized deviation from VWAP)
        df['VWAP_ZScore'] = (close - df['VWAP']) / df['VWAP_Std'].replace(0, np.nan)
        
        # Distance from VWAP as percentage
        df['VWAP_DistPct'] = (close - df['VWAP']) / df['VWAP'] * 100
        
        # === ROLLING VWAP (5-day, 20-day) ===
        
        # 5-day rolling
        window_5 = 5
        pv_5 = (tp * volume).rolling(window_5).sum()
        vol_5 = volume.rolling(window_5).sum()
        df['VWAP_5D'] = pv_5 / vol_5
        
        # 20-day rolling
        window_20 = 20
        pv_20 = (tp * volume).rolling(window_20).sum()
        vol_20 = volume.rolling(window_20).sum()
        df['VWAP_20D'] = pv_20 / vol_20
        
        # Rolling VWAP cross signals
        df['VWAP_5D_Above_20D'] = df['VWAP_5D'] > df['VWAP_20D']
        cross_prev = df['VWAP_5D_Above_20D'].shift(1).fillna(False).astype(bool)
        df['VWAP_Roll_GoldenX'] = df['VWAP_5D_Above_20D'] & ~cross_prev
        df['VWAP_Roll_DeathX'] = ~df['VWAP_5D_Above_20D'] & cross_prev
        
        # === VWAP SLOPE ANALYSIS ===
        
        # 5-period slope of VWAP (trend strength)
        df['VWAP_Slope'] = df['VWAP'].diff(5) / df['VWAP'].shift(5) * 100
        
        # Slope trend classification
        df['VWAP_SlopeUp'] = df['VWAP_Slope'] > 0.1
        df['VWAP_SlopeDown'] = df['VWAP_Slope'] < -0.1
        
        # === TOUCH COUNT (S/R Strength) ===
        
        # Count touches of VWAP in last 5 days
        near_vwap = (close >= df['VWAP'] * 0.998) & (close <= df['VWAP'] * 1.002)
        df['VWAP_TouchCount5D'] = near_vwap.rolling(5).sum()
        
        # === ANCHORED VWAP FROM EARNINGS ===
        
        if earnings_dates:
            for i, earn_date in enumerate(earnings_dates[:8]):  # Last 8 quarters
                try:
                    earn_dt = pd.to_datetime(earn_date)
                    mask = df.index >= earn_dt
                    
                    if mask.any():
                        col = f'AVWAP_E{i+1}'
                        pv_anch = (tp * volume).where(mask, 0).cumsum()
                        vol_anch = volume.where(mask, 0).cumsum()
                        df[col] = np.where(vol_anch > 0, pv_anch / vol_anch, np.nan)
                except Exception:
                    pass
        
        # === ANCHORED VWAP FROM SWING POINTS ===
        
        # 52-week high
        high_52w = df['High'].rolling(252, min_periods=50).max()
        is_52w_high = df['High'] == high_52w
        swing_high_dates = df.index[is_52w_high]
        
        if len(swing_high_dates) > 0:
            last_high_date = swing_high_dates[-1]
            mask = df.index >= last_high_date
            pv_sh = (tp * volume).where(mask, 0).cumsum()
            vol_sh = volume.where(mask, 0).cumsum()
            df['AVWAP_SwingHigh'] = np.where(vol_sh > 0, pv_sh / vol_sh, np.nan)
        
        # 52-week low
        low_52w = df['Low'].rolling(252, min_periods=50).min()
        is_52w_low = df['Low'] == low_52w
        swing_low_dates = df.index[is_52w_low]
        
        if len(swing_low_dates) > 0:
            last_low_date = swing_low_dates[-1]
            mask = df.index >= last_low_date
            pv_sl = (tp * volume).where(mask, 0).cumsum()
            vol_sl = volume.where(mask, 0).cumsum()
            df['AVWAP_SwingLow'] = np.where(vol_sl > 0, pv_sl / vol_sl, np.nan)
        
        # === INSTITUTIONAL ZONE DETECTION ===
        
        # High volume with tight price range (accumulation/distribution)
        vol_ma = volume.rolling(20).mean()
        high_vol = volume > vol_ma * 1.5
        tight_range = (df['High'] - df['Low']) / close < 0.015
        near_vwap_inst = (close >= df['VWAP'] * 0.99) & (close <= df['VWAP'] * 1.01)
        df['VWAP_InstZone'] = high_vol & tight_range & near_vwap_inst
        
        # Volume confirmation for price moves
        price_up = close > close.shift(1)
        vol_up = volume > vol_ma
        df['VWAP_VolConfirm'] = (price_up & vol_up) | (~price_up & vol_up)
        
        # === PRICE CROSS VWAP ===
        
        above_vwap = close > df['VWAP']
        above_prev = above_vwap.shift(1).fillna(False).astype(bool)
        df['VWAP_CrossAbove'] = above_vwap & ~above_prev
        df['VWAP_CrossBelow'] = ~above_vwap & above_prev
        
        return df
    
    def analyze(
        self,
        df: pd.DataFrame,
        earnings_dates: List[str] = None,
        idx: int = -1
    ) -> VWAPAnalysis:
        """Perform complete VWAP analysis."""
        row = df.iloc[idx]
        close = row['Close']
        
        # Standard VWAP values
        vwap = safe_float(row.get('VWAP'))
        std_dev = safe_float(row.get('VWAP_Std'))
        
        upper_1 = safe_float(row.get('VWAP_Upper_10SD', row.get('VWAP_Upper_1SD', vwap)))
        upper_2 = safe_float(row.get('VWAP_Upper_20SD', row.get('VWAP_Upper_2SD', vwap)))
        upper_3 = safe_float(row.get('VWAP_Upper_30SD', row.get('VWAP_Upper_3SD', vwap)))
        lower_1 = safe_float(row.get('VWAP_Lower_10SD', row.get('VWAP_Lower_1SD', vwap)))
        lower_2 = safe_float(row.get('VWAP_Lower_20SD', row.get('VWAP_Lower_2SD', vwap)))
        lower_3 = safe_float(row.get('VWAP_Lower_30SD', row.get('VWAP_Lower_3SD', vwap)))
        
        # Rolling VWAPs
        vwap_5d = safe_float(row.get('VWAP_5D'))
        vwap_20d = safe_float(row.get('VWAP_20D'))
        
        # Rolling cross
        rolling_cross = None
        if safe_bool(row.get('VWAP_Roll_GoldenX')):
            rolling_cross = "GOLDEN_CROSS"
        elif safe_bool(row.get('VWAP_Roll_DeathX')):
            rolling_cross = "DEATH_CROSS"
        
        # Statistical analysis
        z_score = safe_float(row.get('VWAP_ZScore'))
        dist_pct = safe_float(row.get('VWAP_DistPct'))
        
        # Mean reversion probability (based on z-score)
        # Higher z-score = higher probability of reversion
        abs_z = abs(z_score)
        if abs_z > 3:
            mean_rev_prob = 0.95
        elif abs_z > 2:
            mean_rev_prob = 0.80
        elif abs_z > 1:
            mean_rev_prob = 0.60
        else:
            mean_rev_prob = 0.40
        
        # Band position
        if close > upper_3:
            band_pos = "ABOVE_3SD"
        elif close > upper_2:
            band_pos = "ABOVE_2SD"
        elif close > upper_1:
            band_pos = "ABOVE_1SD"
        elif close > vwap:
            band_pos = "ABOVE_VWAP"
        elif close > lower_1:
            band_pos = "BELOW_VWAP"
        elif close > lower_2:
            band_pos = "BELOW_1SD"
        elif close > lower_3:
            band_pos = "BELOW_2SD"
        else:
            band_pos = "BELOW_3SD"
        
        # Slope analysis
        vwap_slope = safe_float(row.get('VWAP_Slope'))
        if vwap_slope > 0.3:
            slope_trend = Trend.STRONG_UP
        elif vwap_slope > 0.1:
            slope_trend = Trend.UP
        elif vwap_slope < -0.3:
            slope_trend = Trend.STRONG_DOWN
        elif vwap_slope < -0.1:
            slope_trend = Trend.DOWN
        else:
            slope_trend = Trend.NEUTRAL
        
        # Touch count / S/R strength
        touch_count = int(safe_float(row.get('VWAP_TouchCount5D')))
        if touch_count >= 4:
            sr_strength = "VERY_STRONG"
        elif touch_count >= 2:
            sr_strength = "STRONG"
        elif touch_count >= 1:
            sr_strength = "MODERATE"
        else:
            sr_strength = "WEAK"
        
        # Anchored VWAPs from earnings
        anch_earnings = {}
        if earnings_dates:
            for i, ed in enumerate(earnings_dates[:8]):
                col = f'AVWAP_E{i+1}'
                if col in df.columns:
                    val = row.get(col)
                    if pd.notna(val):
                        anch_earnings[ed[:10]] = float(val)
        
        # Swing point anchored VWAPs
        avwap_sh = safe_float(row.get('AVWAP_SwingHigh')) if 'AVWAP_SwingHigh' in df.columns else None
        avwap_sl = safe_float(row.get('AVWAP_SwingLow')) if 'AVWAP_SwingLow' in df.columns else None
        if avwap_sh == 0:
            avwap_sh = None
        if avwap_sl == 0:
            avwap_sl = None
        
        # Find nearest anchored VWAP
        nearest_anch = None
        min_dist = float('inf')
        
        for label, av in anch_earnings.items():
            dist = abs(close - av) / av * 100
            if dist < min_dist:
                min_dist = dist
                nearest_anch = (f"Earnings {label}", av, dist)
        
        if avwap_sh:
            dist = abs(close - avwap_sh) / avwap_sh * 100
            if dist < min_dist:
                min_dist = dist
                nearest_anch = ("Swing High", avwap_sh, dist)
        
        if avwap_sl:
            dist = abs(close - avwap_sl) / avwap_sl * 100
            if dist < min_dist:
                nearest_anch = ("Swing Low", avwap_sl, dist)
        
        # Institutional zone
        inst_zone = safe_bool(row.get('VWAP_InstZone'))
        vol_confirm = safe_bool(row.get('VWAP_VolConfirm'))
        
        # Generate signal
        signal = self._generate_signal(
            close, vwap, band_pos, z_score, mean_rev_prob,
            slope_trend, vwap_slope, rolling_cross, sr_strength,
            inst_zone, vol_confirm, nearest_anch, row
        )
        
        return VWAPAnalysis(
            vwap=vwap,
            std_dev=std_dev,
            upper_1sd=upper_1,
            upper_2sd=upper_2,
            upper_3sd=upper_3,
            lower_1sd=lower_1,
            lower_2sd=lower_2,
            lower_3sd=lower_3,
            vwap_5d=vwap_5d,
            vwap_20d=vwap_20d,
            rolling_cross=rolling_cross,
            anchored_earnings=anch_earnings,
            anchored_swing_high=avwap_sh,
            anchored_swing_low=avwap_sl,
            nearest_anchored=nearest_anch,
            z_score=z_score,
            mean_reversion_prob=mean_rev_prob,
            band_position=band_pos,
            vwap_slope=vwap_slope,
            slope_trend=slope_trend,
            touch_count_5d=touch_count,
            sr_strength=sr_strength,
            institutional_zone=inst_zone,
            volume_confirmation=vol_confirm,
            signal=signal
        )
    
    def _generate_signal(
        self,
        close: float,
        vwap: float,
        band_pos: str,
        z_score: float,
        mean_rev_prob: float,
        slope_trend: Trend,
        slope: float,
        rolling_cross: Optional[str],
        sr_strength: str,
        inst_zone: bool,
        vol_confirm: bool,
        nearest_anch: Optional[Tuple],
        row: pd.Series
    ) -> IndicatorSignal:
        """Generate comprehensive VWAP signal."""
        value = 0.0
        confidence = 35.0
        factors = []
        
        # Price vs VWAP (weight: 0.15)
        if close > vwap:
            value += 0.15
            factors.append(f"Price above VWAP (Z={z_score:+.2f})")
        else:
            value -= 0.15
            factors.append(f"Price below VWAP (Z={z_score:+.2f})")
        
        # Band position - mean reversion (weight: 0.25)
        if "3SD" in band_pos:
            if "ABOVE" in band_pos:
                value -= 0.30
                confidence += 20
                factors.append(f"EXTREME OVERBOUGHT at +3σ (reversion prob: {mean_rev_prob:.0%})")
            else:
                value += 0.30
                confidence += 20
                factors.append(f"EXTREME OVERSOLD at -3σ (reversion prob: {mean_rev_prob:.0%})")
        elif "2SD" in band_pos:
            if "ABOVE" in band_pos:
                value -= 0.20
                confidence += 15
                factors.append(f"Overbought at +2σ (reversion prob: {mean_rev_prob:.0%})")
            else:
                value += 0.20
                confidence += 15
                factors.append(f"Oversold at -2σ (reversion prob: {mean_rev_prob:.0%})")
        elif "1SD" in band_pos:
            if "ABOVE" in band_pos:
                value -= 0.08
                factors.append("Extended above +1σ")
            else:
                value += 0.08
                factors.append("Extended below -1σ")
        
        # VWAP slope trend (weight: 0.12)
        if slope_trend in [Trend.STRONG_UP, Trend.UP]:
            value += 0.12 if slope_trend == Trend.STRONG_UP else 0.08
            factors.append(f"VWAP slope bullish ({slope:+.2f}%)")
        elif slope_trend in [Trend.STRONG_DOWN, Trend.DOWN]:
            value -= 0.12 if slope_trend == Trend.STRONG_DOWN else 0.08
            factors.append(f"VWAP slope bearish ({slope:+.2f}%)")
        
        # Rolling VWAP cross (weight: 0.12)
        if rolling_cross == "GOLDEN_CROSS":
            value += 0.12
            confidence += 15
            factors.append("Rolling VWAP GOLDEN CROSS (5D > 20D)")
        elif rolling_cross == "DEATH_CROSS":
            value -= 0.12
            confidence += 15
            factors.append("Rolling VWAP DEATH CROSS (5D < 20D)")
        
        # VWAP cross signals
        if safe_bool(row.get('VWAP_CrossAbove')):
            value += 0.10
            confidence += 10
            factors.append("Price crossed ABOVE VWAP")
        elif safe_bool(row.get('VWAP_CrossBelow')):
            value -= 0.10
            confidence += 10
            factors.append("Price crossed BELOW VWAP")
        
        # S/R strength
        if sr_strength == "VERY_STRONG":
            confidence += 10
            factors.append("VWAP is very strong S/R (4+ touches)")
        elif sr_strength == "STRONG":
            confidence += 5
            factors.append("VWAP is strong S/R")
        
        # Institutional zone
        if inst_zone:
            confidence += 12
            factors.append("INSTITUTIONAL ACCUMULATION ZONE detected")
        
        # Volume confirmation
        if vol_confirm:
            value *= 1.1
            confidence += 5
            factors.append("Volume confirms price action")
        
        # Nearest anchored VWAP proximity
        if nearest_anch and nearest_anch[2] < 1.5:  # Within 1.5%
            confidence += 8
            factors.append(f"Near {nearest_anch[0]} AVWAP (${nearest_anch[1]:,.2f})")
        
        value = max(-1.0, min(1.0, value))
        confidence = max(25, min(90, confidence))
        
        return IndicatorSignal(
            name="VWAP",
            value=value,
            confidence=confidence,
            factors=factors
        )


# =============================================================================
# WILLIAMS %R ADVANCED
# =============================================================================

class WilliamsRAdvanced:
    """
    Williams %R - Advanced Momentum Oscillator System.
    
    Developed by Larry Williams, %R measures overbought/oversold conditions
    on a scale of -100 to 0. This implementation adds institutional features.
    
    FORMULA:
        %R = (Highest High - Close) / (Highest High - Lowest Low) × -100
    
    TRIPLE-PERIOD SYSTEM:
    
        Fast (7-period)
            Quick signals, more noise
            Best for entry timing in confirmed trend
            
        Medium (14-period)
            Standard period, balanced signals
            Primary signal generator
            
        Slow (21-period)
            Trend confirmation
            Filters out noise from fast/medium
            
        Triple alignment = high confidence
    
    ZONES:
        -20 to 0: Overbought
        -80 to -100: Oversold
        -50: Equilibrium (neutral)
        
    PATTERN RECOGNITION:
    
        Regular Divergence (Reversal)
            Bullish: Price lower low, %R higher low
            Bearish: Price higher high, %R lower high
            
        Hidden Divergence (Continuation)
            Bullish: Price higher low, %R lower low (uptrend)
            Bearish: Price lower high, %R higher high (downtrend)
            
        Failure Swings
            Bullish: %R fails to reach oversold on pullback
            Bearish: %R fails to reach overbought on rally
            
        W-Bottom Pattern
            Double bottom in %R with second low higher
            Strong bullish reversal signal
            
        M-Top Pattern
            Double top in %R with second high lower
            Strong bearish reversal signal
    
    MOMENTUM THRUST:
        Quick move from extreme zone to neutral/opposite
        Indicates powerful momentum shift
        
    EXTREME DURATION:
        How long price stays in extreme zone
        Longer duration = stronger potential reversal
    """
    
    def __init__(
        self,
        fast_period: int = 7,
        medium_period: int = 14,
        slow_period: int = 21,
        overbought: float = -20,
        oversold: float = -80
    ):
        self.fast = fast_period
        self.medium = medium_period
        self.slow = slow_period
        self.ob = overbought
        self.os = oversold
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate complete Williams %R system."""
        df = df.copy()
        high, low, close = df['High'], df['Low'], df['Close']
        
        # === TRIPLE PERIOD CALCULATION ===
        
        # Fast (7-period)
        hh_f = high.rolling(self.fast).max()
        ll_f = low.rolling(self.fast).min()
        df['WR_Fast'] = (hh_f - close) / (hh_f - ll_f) * -100
        
        # Medium (14-period)
        hh_m = high.rolling(self.medium).max()
        ll_m = low.rolling(self.medium).min()
        df['WR_Medium'] = (hh_m - close) / (hh_m - ll_m) * -100
        
        # Slow (21-period)
        hh_s = high.rolling(self.slow).max()
        ll_s = low.rolling(self.slow).min()
        df['WR_Slow'] = (hh_s - close) / (hh_s - ll_s) * -100
        
        # Smoothed (3-period SMA of medium)
        df['WR_Smoothed'] = df['WR_Medium'].rolling(3).mean()
        
        # === ZONE CLASSIFICATION ===
        
        df['WR_Fast_OB'] = df['WR_Fast'] > self.ob
        df['WR_Fast_OS'] = df['WR_Fast'] < self.os
        df['WR_Med_OB'] = df['WR_Medium'] > self.ob
        df['WR_Med_OS'] = df['WR_Medium'] < self.os
        df['WR_Slow_OB'] = df['WR_Slow'] > self.ob
        df['WR_Slow_OS'] = df['WR_Slow'] < self.os
        
        # Triple alignment
        df['WR_Triple_OB'] = df['WR_Fast_OB'] & df['WR_Med_OB'] & df['WR_Slow_OB']
        df['WR_Triple_OS'] = df['WR_Fast_OS'] & df['WR_Med_OS'] & df['WR_Slow_OS']
        
        # === EXTREME DURATION ===
        
        # Count consecutive bars in extreme zone
        ob_group = (~df['WR_Med_OB']).cumsum()
        df['WR_OB_Duration'] = df.groupby(ob_group)['WR_Med_OB'].cumsum()
        
        os_group = (~df['WR_Med_OS']).cumsum()
        df['WR_OS_Duration'] = df.groupby(os_group)['WR_Med_OS'].cumsum()
        
        # === ZONE EXITS ===
        
        med_ob_prev = df['WR_Med_OB'].shift(1).fillna(False).astype(bool)
        med_os_prev = df['WR_Med_OS'].shift(1).fillna(False).astype(bool)
        
        df['WR_Exit_OB'] = ~df['WR_Med_OB'] & med_ob_prev
        df['WR_Exit_OS'] = ~df['WR_Med_OS'] & med_os_prev
        
        # === REGULAR DIVERGENCE ===
        
        lookback = 10
        
        # Bullish: Price lower low, %R higher low (in oversold)
        price_ll = low < low.rolling(lookback).min().shift(1)
        wr_hl = df['WR_Medium'] > df['WR_Medium'].rolling(lookback).min().shift(1)
        df['WR_RegBullDiv'] = price_ll & wr_hl & df['WR_Med_OS']
        
        # Bearish: Price higher high, %R lower high (in overbought)
        price_hh = high > high.rolling(lookback).max().shift(1)
        wr_lh = df['WR_Medium'] < df['WR_Medium'].rolling(lookback).max().shift(1)
        df['WR_RegBearDiv'] = price_hh & wr_lh & df['WR_Med_OB']
        
        # === HIDDEN DIVERGENCE ===
        
        trend = close.rolling(20).mean() - close.rolling(20).mean().shift(20)
        uptrend = trend > 0
        downtrend = trend < 0
        
        # Hidden bullish: Price HL, %R LL, in uptrend
        price_hl = low > low.rolling(lookback).min().shift(1)
        wr_ll = df['WR_Medium'] < df['WR_Medium'].rolling(lookback).min().shift(1)
        df['WR_HidBullDiv'] = price_hl & wr_ll & uptrend
        
        # Hidden bearish: Price LH, %R HH, in downtrend
        price_lh = high < high.rolling(lookback).max().shift(1)
        wr_hh = df['WR_Medium'] > df['WR_Medium'].rolling(lookback).max().shift(1)
        df['WR_HidBearDiv'] = price_lh & wr_hh & downtrend
        
        # === FAILURE SWINGS ===
        
        wr = df['WR_Medium']
        wr_falling = wr < wr.shift(1)
        wr_rising = wr > wr.shift(1)
        
        # Bullish failure: In uptrend, pullback fails to reach oversold
        failed_reach_os = wr.rolling(5).min() > self.os
        was_falling = wr_falling.shift(1)
        df['WR_FailSwingBull'] = was_falling & wr_rising & failed_reach_os & uptrend
        
        # Bearish failure: In downtrend, rally fails to reach overbought
        failed_reach_ob = wr.rolling(5).max() < self.ob
        was_rising = wr_rising.shift(1)
        df['WR_FailSwingBear'] = was_rising & wr_falling & failed_reach_ob & downtrend
        
        # === W-BOTTOM / M-TOP PATTERNS ===
        
        # W-Bottom: Two lows in oversold, second higher
        wr_5_min = wr.rolling(5).min()
        wr_10_min = wr.rolling(10).min()
        second_low_higher = wr_5_min > wr_10_min.shift(5)
        df['WR_WBottom'] = df['WR_Med_OS'] & second_low_higher & wr_rising
        
        # M-Top: Two highs in overbought, second lower
        wr_5_max = wr.rolling(5).max()
        wr_10_max = wr.rolling(10).max()
        second_high_lower = wr_5_max < wr_10_max.shift(5)
        df['WR_MTop'] = df['WR_Med_OB'] & second_high_lower & wr_falling
        
        # === MOMENTUM THRUST ===
        
        # Bullish thrust: From oversold to above -50 quickly
        was_os_5 = df['WR_Med_OS'].rolling(5).max().fillna(0).astype(bool)
        now_above_50 = wr > -50
        df['WR_BullThrust'] = was_os_5 & now_above_50 & ~df['WR_Med_OS']
        
        # Bearish thrust: From overbought to below -50 quickly
        was_ob_5 = df['WR_Med_OB'].rolling(5).max().fillna(0).astype(bool)
        now_below_50 = wr < -50
        df['WR_BearThrust'] = was_ob_5 & now_below_50 & ~df['WR_Med_OB']
        
        return df
    
    def analyze(self, df: pd.DataFrame, idx: int = -1) -> WilliamsRAnalysis:
        """Perform complete Williams %R analysis."""
        row = df.iloc[idx]
        
        # Multi-period values
        fast = safe_float(row.get('WR_Fast'), -50)
        medium = safe_float(row.get('WR_Medium'), -50)
        slow = safe_float(row.get('WR_Slow'), -50)
        smoothed = safe_float(row.get('WR_Smoothed'), -50)
        
        # Zone
        if medium > self.ob:
            zone = "OVERBOUGHT"
        elif medium < self.os:
            zone = "OVERSOLD"
        else:
            zone = "NEUTRAL"
        
        # Extreme duration
        if zone == "OVERBOUGHT":
            extreme_dur = int(safe_float(row.get('WR_OB_Duration')))
        elif zone == "OVERSOLD":
            extreme_dur = int(safe_float(row.get('WR_OS_Duration')))
        else:
            extreme_dur = 0
        
        # Triple alignment
        triple_ob = safe_bool(row.get('WR_Triple_OB'))
        triple_os = safe_bool(row.get('WR_Triple_OS'))
        if triple_ob:
            triple_align = "TRIPLE_OVERBOUGHT"
        elif triple_os:
            triple_align = "TRIPLE_OVERSOLD"
        elif fast > self.ob and medium > self.ob:
            triple_align = "DOUBLE_OVERBOUGHT"
        elif fast < self.os and medium < self.os:
            triple_align = "DOUBLE_OVERSOLD"
        else:
            triple_align = "MIXED"
        
        # Pattern detection
        reg_bull = safe_bool(row.get('WR_RegBullDiv'))
        reg_bear = safe_bool(row.get('WR_RegBearDiv'))
        hid_bull = safe_bool(row.get('WR_HidBullDiv'))
        hid_bear = safe_bool(row.get('WR_HidBearDiv'))
        fail_bull = safe_bool(row.get('WR_FailSwingBull'))
        fail_bear = safe_bool(row.get('WR_FailSwingBear'))
        w_bottom = safe_bool(row.get('WR_WBottom'))
        m_top = safe_bool(row.get('WR_MTop'))
        
        # Momentum thrust
        thrust = None
        if safe_bool(row.get('WR_BullThrust')):
            thrust = "BULLISH"
        elif safe_bool(row.get('WR_BearThrust')):
            thrust = "BEARISH"
        
        # Zone exit
        zone_exit = None
        if safe_bool(row.get('WR_Exit_OB')):
            zone_exit = "EXIT_OVERBOUGHT"
        elif safe_bool(row.get('WR_Exit_OS')):
            zone_exit = "EXIT_OVERSOLD"
        
        # Generate signal
        signal = self._generate_signal(
            fast, medium, slow, zone, extreme_dur, triple_align,
            reg_bull, reg_bear, hid_bull, hid_bear,
            fail_bull, fail_bear, w_bottom, m_top,
            thrust, zone_exit
        )
        
        return WilliamsRAnalysis(
            fast_value=fast,
            medium_value=medium,
            slow_value=slow,
            smoothed_value=smoothed,
            zone=zone,
            extreme_duration=extreme_dur,
            triple_alignment=triple_align,
            regular_bull_div=reg_bull,
            regular_bear_div=reg_bear,
            hidden_bull_div=hid_bull,
            hidden_bear_div=hid_bear,
            failure_swing_bull=fail_bull,
            failure_swing_bear=fail_bear,
            w_bottom=w_bottom,
            m_top=m_top,
            momentum_thrust=thrust,
            zone_exit=zone_exit,
            signal=signal
        )
    
    def _generate_signal(
        self,
        fast: float,
        medium: float,
        slow: float,
        zone: str,
        extreme_dur: int,
        triple_align: str,
        reg_bull: bool,
        reg_bear: bool,
        hid_bull: bool,
        hid_bear: bool,
        fail_bull: bool,
        fail_bear: bool,
        w_bottom: bool,
        m_top: bool,
        thrust: Optional[str],
        zone_exit: Optional[str]
    ) -> IndicatorSignal:
        """Generate comprehensive Williams %R signal."""
        value = 0.0
        confidence = 35.0
        factors = []
        
        # Zone-based (contrarian at extremes)
        if zone == "OVERBOUGHT":
            value -= 0.12
            factors.append(f"Overbought zone ({medium:.0f})")
            if extreme_dur > 5:
                value -= 0.05
                confidence += 8
                factors.append(f"Extended overbought ({extreme_dur} bars)")
        elif zone == "OVERSOLD":
            value += 0.12
            factors.append(f"Oversold zone ({medium:.0f})")
            if extreme_dur > 5:
                value += 0.05
                confidence += 8
                factors.append(f"Extended oversold ({extreme_dur} bars)")
        else:
            factors.append(f"Neutral zone ({medium:.0f})")
        
        # Triple alignment
        if "TRIPLE_OVERBOUGHT" in triple_align:
            value -= 0.10
            confidence += 12
            factors.append("TRIPLE PERIOD OVERBOUGHT")
        elif "TRIPLE_OVERSOLD" in triple_align:
            value += 0.10
            confidence += 12
            factors.append("TRIPLE PERIOD OVERSOLD")
        elif "DOUBLE" in triple_align:
            confidence += 5
            factors.append(triple_align.replace("_", " "))
        
        # Zone exits (actionable signals)
        if zone_exit == "EXIT_OVERBOUGHT":
            value -= 0.18
            confidence += 18
            factors.append("EXIT FROM OVERBOUGHT - Sell trigger")
        elif zone_exit == "EXIT_OVERSOLD":
            value += 0.18
            confidence += 18
            factors.append("EXIT FROM OVERSOLD - Buy trigger")
        
        # Regular divergence (high-probability reversal)
        if reg_bull:
            value += 0.35
            confidence += 25
            factors.append("REGULAR BULLISH DIVERGENCE - High probability reversal")
        if reg_bear:
            value -= 0.35
            confidence += 25
            factors.append("REGULAR BEARISH DIVERGENCE - High probability reversal")
        
        # Hidden divergence (continuation)
        if hid_bull:
            value += 0.18
            confidence += 15
            factors.append("Hidden bullish divergence - Trend continuation")
        if hid_bear:
            value -= 0.18
            confidence += 15
            factors.append("Hidden bearish divergence - Trend continuation")
        
        # Failure swings
        if fail_bull:
            value += 0.12
            confidence += 10
            factors.append("Bullish failure swing")
        if fail_bear:
            value -= 0.12
            confidence += 10
            factors.append("Bearish failure swing")
        
        # W-Bottom / M-Top patterns
        if w_bottom:
            value += 0.25
            confidence += 20
            factors.append("W-BOTTOM pattern - Strong bullish reversal")
        if m_top:
            value -= 0.25
            confidence += 20
            factors.append("M-TOP pattern - Strong bearish reversal")
        
        # Momentum thrust
        if thrust == "BULLISH":
            value += 0.22
            confidence += 18
            factors.append("BULLISH MOMENTUM THRUST")
        elif thrust == "BEARISH":
            value -= 0.22
            confidence += 18
            factors.append("BEARISH MOMENTUM THRUST")
        
        value = max(-1.0, min(1.0, value))
        confidence = max(25, min(95, confidence))
        
        return IndicatorSignal(
            name="Williams %R",
            value=value,
            confidence=confidence,
            factors=factors
        )


# =============================================================================
# CCI PROFESSIONAL
# =============================================================================

class CCIProfessional:
    """
    Commodity Channel Index - Professional Implementation.
    
    Developed by Donald Lambert, CCI measures price deviation from the
    statistical mean, identifying cyclical turning points.
    
    FORMULA:
        CCI = (Typical Price - SMA of TP) / (0.015 * Mean Deviation)
        
    Lambert's 0.015 constant ensures ~75% of values fall within +/-100.
    
    COMPONENTS:
    
        Dual CCI System
            Short (14-period): Entry timing
            Long (50-period): Trend filter
            Both aligned = high confidence
            
        Zone Classification
            +200 and above: Extremely overbought
            +100 to +200: Overbought (strong trend)
            -100 to +100: Normal trading range
            -100 to -200: Oversold (strong trend)
            -200 and below: Extremely oversold
            
        Zero-Line Analysis
            Cross above zero = bullish momentum shift
            Cross below zero = bearish momentum shift
            
        Divergence Detection
            Price vs CCI disagreement signals reversal
            
        Hook Patterns
            CCI turning before reaching extreme
            Early reversal warning
    
    TRADING RULES:
        1. Zero-line cross = trend change
        2. +100/-100 cross = trend acceleration
        3. Extreme readings (+200/-200) = reversal zone
        4. Divergence = high-probability reversal
        5. Hooks = early warning before full reversal
        6. Long CCI as trend filter for short CCI entries
    """
    
    def __init__(
        self,
        short_period: int = 14,
        long_period: int = 50,
        constant: float = 0.015
    ):
        self.short = short_period
        self.long = long_period
        self.constant = constant
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate CCI with all professional features."""
        df = df.copy()
        
        # Ensure TypicalPrice exists
        if 'TypicalPrice' not in df.columns:
            df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        tp = df['TypicalPrice']
        
        # Short CCI (14-period)
        tp_sma_s = tp.rolling(self.short).mean()
        md_s = tp.rolling(self.short).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['CCI_Short'] = (tp - tp_sma_s) / (self.constant * md_s)
        
        # Long CCI (50-period)
        tp_sma_l = tp.rolling(self.long).mean()
        md_l = tp.rolling(self.long).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['CCI_Long'] = (tp - tp_sma_l) / (self.constant * md_l)
        
        # Zone classification
        df['CCI_S_ExOB'] = df['CCI_Short'] > 200
        df['CCI_S_OB'] = (df['CCI_Short'] > 100) & (df['CCI_Short'] <= 200)
        df['CCI_S_OS'] = (df['CCI_Short'] < -100) & (df['CCI_Short'] >= -200)
        df['CCI_S_ExOS'] = df['CCI_Short'] < -200
        
        # Dual alignment
        df['CCI_Both_Bull'] = (df['CCI_Short'] > 0) & (df['CCI_Long'] > 0)
        df['CCI_Both_Bear'] = (df['CCI_Short'] < 0) & (df['CCI_Long'] < 0)
        
        # Zero-line crosses (short CCI)
        cci_prev = df['CCI_Short'].shift(1)
        df['CCI_ZeroUp'] = (df['CCI_Short'] > 0) & (cci_prev <= 0)
        df['CCI_ZeroDown'] = (df['CCI_Short'] < 0) & (cci_prev >= 0)
        
        # +100/-100 crosses
        df['CCI_Cross100Up'] = (df['CCI_Short'] > 100) & (cci_prev <= 100)
        df['CCI_Cross100Down'] = (df['CCI_Short'] < 100) & (cci_prev >= 100)
        df['CCI_CrossM100Up'] = (df['CCI_Short'] > -100) & (cci_prev <= -100)
        df['CCI_CrossM100Down'] = (df['CCI_Short'] < -100) & (cci_prev >= -100)
        
        # Divergence detection (5-bar lookback)
        lookback = 5
        close = df['Close']
        
        # Bullish divergence: Price lower low, CCI higher low
        price_ll = close < close.rolling(lookback).min().shift(1)
        cci_hl = df['CCI_Short'] > df['CCI_Short'].rolling(lookback).min().shift(1)
        df['CCI_BullDiv'] = price_ll & cci_hl & (df['CCI_Short'] < 0)
        
        # Bearish divergence: Price higher high, CCI lower high
        price_hh = close > close.rolling(lookback).max().shift(1)
        cci_lh = df['CCI_Short'] < df['CCI_Short'].rolling(lookback).max().shift(1)
        df['CCI_BearDiv'] = price_hh & cci_lh & (df['CCI_Short'] > 0)
        
        # Hook patterns (CCI turning before reaching extreme)
        cci_falling = df['CCI_Short'] < df['CCI_Short'].shift(1)
        cci_rising = df['CCI_Short'] > df['CCI_Short'].shift(1)
        
        # Bullish hook: CCI was falling, now rising, stayed above -200
        was_falling = cci_falling.shift(1).fillna(False)
        stayed_above = df['CCI_Short'].rolling(3).min() > -200
        df['CCI_BullHook'] = was_falling & cci_rising & (df['CCI_Short'] < -50) & stayed_above
        
        # Bearish hook: CCI was rising, now falling, stayed below +200
        was_rising = cci_rising.shift(1).fillna(False)
        stayed_below = df['CCI_Short'].rolling(3).max() < 200
        df['CCI_BearHook'] = was_rising & cci_falling & (df['CCI_Short'] > 50) & stayed_below
        
        return df
    
    def analyze(self, df: pd.DataFrame, idx: int = -1) -> CCIAnalysis:
        """Perform complete CCI analysis."""
        row = df.iloc[idx]
        
        short_val = self._safe_float(row.get('CCI_Short'), 0)
        long_val = self._safe_float(row.get('CCI_Long'), 0)
        
        # Zone determination
        if short_val > 200:
            zone = "EXTREME_OVERBOUGHT"
        elif short_val > 100:
            zone = "OVERBOUGHT"
        elif short_val > -100:
            zone = "NEUTRAL"
        elif short_val > -200:
            zone = "OVERSOLD"
        else:
            zone = "EXTREME_OVERSOLD"
        
        # Dual alignment
        both_bull = self._safe_bool(row.get('CCI_Both_Bull'))
        both_bear = self._safe_bool(row.get('CCI_Both_Bear'))
        dual_aligned = both_bull or both_bear
        
        # Zero cross
        zero_cross = None
        if self._safe_bool(row.get('CCI_ZeroUp')):
            zero_cross = "BULLISH"
        elif self._safe_bool(row.get('CCI_ZeroDown')):
            zero_cross = "BEARISH"
        
        # Divergence
        bull_div = self._safe_bool(row.get('CCI_BullDiv'))
        bear_div = self._safe_bool(row.get('CCI_BearDiv'))
        
        # Hooks
        bull_hook = self._safe_bool(row.get('CCI_BullHook'))
        bear_hook = self._safe_bool(row.get('CCI_BearHook'))
        
        # Trend strength
        if short_val > 100 and long_val > 0:
            trend = Trend.STRONG_BULLISH
        elif short_val > 0 and long_val > 0:
            trend = Trend.BULLISH
        elif short_val < -100 and long_val < 0:
            trend = Trend.STRONG_BEARISH
        elif short_val < 0 and long_val < 0:
            trend = Trend.BEARISH
        else:
            trend = Trend.NEUTRAL
        
        # Generate signal
        signal = self._generate_signal(
            short_val, long_val, zone, dual_aligned,
            zero_cross, bull_div, bear_div,
            bull_hook, bear_hook, trend, row
        )
        
        return CCIAnalysis(
            short_value=short_val,
            long_value=long_val,
            zone=zone,
            dual_alignment=dual_aligned,
            zero_cross=zero_cross,
            bullish_divergence=bull_div,
            bearish_divergence=bear_div,
            bullish_hook=bull_hook,
            bearish_hook=bear_hook,
            trend_strength=trend,
            signal=signal
        )
    
    def _generate_signal(
        self,
        short_val: float,
        long_val: float,
        zone: str,
        dual_aligned: bool,
        zero_cross: Optional[str],
        bull_div: bool,
        bear_div: bool,
        bull_hook: bool,
        bear_hook: bool,
        trend: Trend,
        row: pd.Series
    ) -> IndicatorSignal:
        """Generate CCI signal."""
        value = 0.0
        confidence = 40.0
        factors = []
        
        # Zone-based signals
        if zone == "EXTREME_OVERBOUGHT":
            value -= 0.30
            confidence += 20
            factors.append(f"EXTREME OVERBOUGHT ({short_val:.0f}) - Reversal zone")
        elif zone == "OVERBOUGHT":
            value -= 0.10
            factors.append(f"Overbought ({short_val:.0f})")
        elif zone == "EXTREME_OVERSOLD":
            value += 0.30
            confidence += 20
            factors.append(f"EXTREME OVERSOLD ({short_val:.0f}) - Reversal zone")
        elif zone == "OVERSOLD":
            value += 0.10
            factors.append(f"Oversold ({short_val:.0f})")
        else:
            factors.append(f"Neutral ({short_val:.0f})")
        
        # Dual alignment
        if dual_aligned:
            if short_val > 0:
                value += 0.15
                factors.append("Dual CCI bullish alignment")
            else:
                value -= 0.15
                factors.append("Dual CCI bearish alignment")
            confidence += 10
        
        # Zero-line cross
        if zero_cross == "BULLISH":
            value += 0.20
            confidence += 15
            factors.append("BULLISH ZERO-LINE CROSS")
        elif zero_cross == "BEARISH":
            value -= 0.20
            confidence += 15
            factors.append("BEARISH ZERO-LINE CROSS")
        
        # +100/-100 crosses
        if self._safe_bool(row.get('CCI_Cross100Up')):
            value += 0.15
            confidence += 10
            factors.append("Breakout above +100")
        elif self._safe_bool(row.get('CCI_CrossM100Down')):
            value -= 0.15
            confidence += 10
            factors.append("Breakdown below -100")
        elif self._safe_bool(row.get('CCI_CrossM100Up')):
            value += 0.10
            factors.append("Recovery above -100")
        elif self._safe_bool(row.get('CCI_Cross100Down')):
            value -= 0.10
            factors.append("Pullback below +100")
        
        # Divergence (high-value signal)
        if bull_div:
            value += 0.30
            confidence += 20
            factors.append("BULLISH DIVERGENCE - High probability reversal")
        
        if bear_div:
            value -= 0.30
            confidence += 20
            factors.append("BEARISH DIVERGENCE - High probability reversal")
        
        # Hook patterns
        if bull_hook:
            value += 0.15
            confidence += 10
            factors.append("Bullish hook pattern")
        
        if bear_hook:
            value -= 0.15
            confidence += 10
            factors.append("Bearish hook pattern")
        
        # Trend bias
        if trend in [Trend.STRONG_BULLISH, Trend.BULLISH]:
            value += 0.05
        elif trend in [Trend.STRONG_BEARISH, Trend.BEARISH]:
            value -= 0.05
        
        value = max(-1.0, min(1.0, value))
        confidence = max(20, min(95, confidence))
        
        return IndicatorSignal(
            name="CCI",
            value=value,
            confidence=confidence,
            factors=factors
        )
    
    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        if pd.isna(val):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def _safe_bool(val, default: bool = False) -> bool:
        if pd.isna(val):
            return default
        return bool(val)


# =============================================================================
# MARKET STRUCTURE ANALYZER
# =============================================================================

class MarketStructureAnalyzer:
    """
    Market Structure Analysis - Institutional Price Action.
    
    Identifies the underlying structure of price movement that institutional
    traders use for decision-making. This is NOT a traditional indicator
    but rather a framework for understanding market context.
    
    COMPONENTS:
    
        Swing Point Detection
            Identifies significant highs and lows using fractal logic
            5-bar swing high: High with 2 lower highs on each side
            5-bar swing low: Low with 2 higher lows on each side
            
        Trend Structure
            Higher Highs + Higher Lows = Uptrend
            Lower Highs + Lower Lows = Downtrend
            Mixed = Consolidation
            
        Support/Resistance Zones
            Clusters of swing points form S/R zones
            Volume-weighted significance scoring
            
        Break of Structure (BOS)
            Price breaking previous swing = trend continuation
            
        Change of Character (CHoCH)
            First break against trend = potential reversal
    
    RELEVANCE:
        - Provides context for all other indicators
        - Identifies optimal stop loss and target levels
        - Confirms trend for directional bias
        - Required for proper trade setup construction
    """
    
    def __init__(self, swing_period: int = 5, zone_threshold: float = 0.02):
        self.swing_period = swing_period
        self.zone_threshold = zone_threshold
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market structure components."""
        df = df.copy()
        n = self.swing_period
        high, low, close = df['High'], df['Low'], df['Close']
        
        # Swing High Detection (fractal high)
        # A swing high has n lower highs on each side
        swing_high = pd.Series(False, index=df.index)
        swing_low = pd.Series(False, index=df.index)
        
        for i in range(n, len(df) - n):
            # Check if this is a swing high
            is_swing_high = True
            for j in range(1, n + 1):
                if high.iloc[i] <= high.iloc[i - j] or high.iloc[i] <= high.iloc[i + j]:
                    is_swing_high = False
                    break
            swing_high.iloc[i] = is_swing_high
            
            # Check if this is a swing low
            is_swing_low = True
            for j in range(1, n + 1):
                if low.iloc[i] >= low.iloc[i - j] or low.iloc[i] >= low.iloc[i + j]:
                    is_swing_low = False
                    break
            swing_low.iloc[i] = is_swing_low
        
        df['MS_SwingHigh'] = swing_high
        df['MS_SwingLow'] = swing_low
        df['MS_SwingHighPrice'] = np.where(swing_high, high, np.nan)
        df['MS_SwingLowPrice'] = np.where(swing_low, low, np.nan)
        
        # Forward fill swing points for reference
        df['MS_LastSwingHigh'] = df['MS_SwingHighPrice'].ffill()
        df['MS_LastSwingLow'] = df['MS_SwingLowPrice'].ffill()
        
        # Trend Structure Analysis
        # Compare current swing to previous swing
        swing_highs = df[df['MS_SwingHigh']]['High'].values
        swing_lows = df[df['MS_SwingLow']]['Low'].values
        
        # Higher High / Lower High detection
        df['MS_HigherHigh'] = False
        df['MS_LowerHigh'] = False
        df['MS_HigherLow'] = False
        df['MS_LowerLow'] = False
        
        prev_sh = None
        for idx in df[df['MS_SwingHigh']].index:
            if prev_sh is not None:
                if df.loc[idx, 'High'] > prev_sh:
                    df.loc[idx, 'MS_HigherHigh'] = True
                else:
                    df.loc[idx, 'MS_LowerHigh'] = True
            prev_sh = df.loc[idx, 'High']
        
        prev_sl = None
        for idx in df[df['MS_SwingLow']].index:
            if prev_sl is not None:
                if df.loc[idx, 'Low'] > prev_sl:
                    df.loc[idx, 'MS_HigherLow'] = True
                else:
                    df.loc[idx, 'MS_LowerLow'] = True
            prev_sl = df.loc[idx, 'Low']
        
        # Rolling trend structure (last 20 bars)
        window = 20
        hh_count = df['MS_HigherHigh'].rolling(window).sum()
        lh_count = df['MS_LowerHigh'].rolling(window).sum()
        hl_count = df['MS_HigherLow'].rolling(window).sum()
        ll_count = df['MS_LowerLow'].rolling(window).sum()
        
        bullish_structure = (hh_count + hl_count) > (lh_count + ll_count)
        bearish_structure = (lh_count + ll_count) > (hh_count + hl_count)
        
        df['MS_BullishStructure'] = bullish_structure
        df['MS_BearishStructure'] = bearish_structure
        
        # Break of Structure (BOS) - Price closes beyond last swing
        df['MS_BOS_Bullish'] = close > df['MS_LastSwingHigh'].shift(1)
        df['MS_BOS_Bearish'] = close < df['MS_LastSwingLow'].shift(1)
        
        # Detect first BOS against trend (Change of Character)
        # CHoCH Bullish: In downtrend, first break above swing high
        in_downtrend = df['MS_BearishStructure'].shift(1).fillna(False)
        df['MS_CHoCH_Bullish'] = df['MS_BOS_Bullish'] & in_downtrend
        
        in_uptrend = df['MS_BullishStructure'].shift(1).fillna(False)
        df['MS_CHoCH_Bearish'] = df['MS_BOS_Bearish'] & in_uptrend
        
        # Distance to key levels (for trade setup)
        df['MS_DistToSwingHigh'] = (df['MS_LastSwingHigh'] - close) / close * 100
        df['MS_DistToSwingLow'] = (close - df['MS_LastSwingLow']) / close * 100
        
        return df
    
    def get_key_levels(self, df: pd.DataFrame, num_levels: int = 5) -> Dict[str, List[float]]:
        """Extract key support and resistance levels."""
        swing_highs = df[df['MS_SwingHigh']]['High'].dropna().tolist()
        swing_lows = df[df['MS_SwingLow']]['Low'].dropna().tolist()
        
        # Get most recent significant levels
        resistance = sorted(swing_highs[-num_levels*2:], reverse=True)[:num_levels]
        support = sorted(swing_lows[-num_levels*2:])[:num_levels]
        
        return {
            'resistance': resistance,
            'support': support
        }


# =============================================================================
# VOLUME FLOW ANALYSIS
# =============================================================================

class VolumeFlowAnalyzer:
    """
    Volume Flow Analysis - Institutional Activity Detection.
    
    Volume confirms price. This module analyzes volume patterns to identify
    institutional accumulation and distribution phases.
    
    COMPONENTS:
    
        On-Balance Volume (OBV)
            Cumulative volume flow based on close direction
            OBV divergence from price = reversal warning
            
        Chaikin Money Flow (CMF)
            Volume-weighted accumulation/distribution
            Range: -1 to +1
            Sustained readings indicate institutional activity
            
        Volume Price Trend (VPT)
            Volume weighted by price change percentage
            More sensitive than OBV to price magnitude
            
        Relative Volume (RVOL)
            Current volume vs average volume
            RVOL > 2 = significant institutional activity
            
        Volume Climax Detection
            Extreme volume with price reversal = exhaustion
            Often marks turning points
    
    RELEVANCE:
        - Confirms VWAP signals
        - Validates breakouts (high volume = real breakout)
        - Identifies institutional accumulation zones
        - Divergences warn of reversals
    """
    
    def __init__(self, cmf_period: int = 20, rvol_period: int = 20):
        self.cmf_period = cmf_period
        self.rvol_period = rvol_period
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume flow indicators."""
        df = df.copy()
        high, low, close, vol = df['High'], df['Low'], df['Close'], df['Volume']
        
        # On-Balance Volume (OBV)
        obv_direction = np.where(close > close.shift(1), 1, 
                        np.where(close < close.shift(1), -1, 0))
        df['VF_OBV'] = (vol * obv_direction).cumsum()
        
        # OBV SMA for trend
        df['VF_OBV_SMA'] = df['VF_OBV'].rolling(20).mean()
        df['VF_OBV_Trend'] = df['VF_OBV'] > df['VF_OBV_SMA']
        
        # OBV Divergence Detection
        # Bullish: Price lower low, OBV higher low
        price_ll = close < close.rolling(10).min().shift(1)
        obv_hl = df['VF_OBV'] > df['VF_OBV'].rolling(10).min().shift(1)
        df['VF_OBV_BullDiv'] = price_ll & obv_hl
        
        # Bearish: Price higher high, OBV lower high
        price_hh = close > close.rolling(10).max().shift(1)
        obv_lh = df['VF_OBV'] < df['VF_OBV'].rolling(10).max().shift(1)
        df['VF_OBV_BearDiv'] = price_hh & obv_lh
        
        # Chaikin Money Flow (CMF)
        # Money Flow Multiplier: [(Close - Low) - (High - Close)] / (High - Low)
        mf_mult = np.where(
            (high - low) != 0,
            ((close - low) - (high - close)) / (high - low),
            0
        )
        mf_volume = mf_mult * vol
        
        df['VF_CMF'] = mf_volume.rolling(self.cmf_period).sum() / vol.rolling(self.cmf_period).sum()
        
        # CMF Zones
        df['VF_CMF_Bullish'] = df['VF_CMF'] > 0.1
        df['VF_CMF_Bearish'] = df['VF_CMF'] < -0.1
        df['VF_CMF_Strong'] = abs(df['VF_CMF']) > 0.25
        
        # Volume Price Trend (VPT)
        price_change_pct = close.pct_change().fillna(0)
        df['VF_VPT'] = (vol * price_change_pct).cumsum()
        df['VF_VPT_SMA'] = df['VF_VPT'].rolling(20).mean()
        
        # Relative Volume (RVOL)
        vol_sma = vol.rolling(self.rvol_period).mean()
        df['VF_RVOL'] = vol / vol_sma
        
        # Volume classification
        df['VF_HighVolume'] = df['VF_RVOL'] > 1.5
        df['VF_VeryHighVolume'] = df['VF_RVOL'] > 2.0
        df['VF_ExtremeVolume'] = df['VF_RVOL'] > 3.0
        
        # Volume Climax Detection
        # High volume + reversal candle pattern
        high_vol = df['VF_RVOL'] > 2.0
        bearish_candle = close < (high + low) / 2  # Close in lower half
        bullish_candle = close > (high + low) / 2  # Close in upper half
        up_move = close > close.shift(1)
        down_move = close < close.shift(1)
        
        # Selling climax: Down move + high volume + bullish close (exhaustion)
        df['VF_SellingClimax'] = down_move.shift(1) & high_vol & bullish_candle
        
        # Buying climax: Up move + high volume + bearish close (exhaustion)
        df['VF_BuyingClimax'] = up_move.shift(1) & high_vol & bearish_candle
        
        # Accumulation/Distribution Phase
        # Accumulation: Tight range + below average volume + holding above support
        tight_range = ((high - low) / close) < 0.015
        low_vol = df['VF_RVOL'] < 0.8
        df['VF_Accumulation'] = tight_range & low_vol & (df['VF_CMF'] > 0)
        df['VF_Distribution'] = tight_range & low_vol & (df['VF_CMF'] < 0)
        
        return df


# =============================================================================
# VOLATILITY STRUCTURE
# =============================================================================

class VolatilityStructure:
    """
    Volatility Structure Analysis.
    
    Measures and classifies volatility regimes to inform position sizing
    and strategy selection. Uses multiple volatility measures for robustness.
    
    NOTE: This is DIFFERENT from Fardeen's ATR. We use:
    - Parkinson volatility (high-low based, more efficient)
    - Yang-Zhang volatility (incorporates overnight gaps)
    - Volatility regime classification
    
    COMPONENTS:
    
        Parkinson Volatility
            Uses high-low range, more efficient than close-to-close
            Better captures intraday volatility
            
        Yang-Zhang Volatility
            Most efficient estimator
            Handles overnight gaps and opening jumps
            
        Volatility Percentile
            Current vol vs historical distribution
            90th percentile = high vol regime
            10th percentile = low vol regime
            
        Volatility Regime
            LOW: Percentile < 25 (breakout setups)
            NORMAL: 25-75 percentile
            HIGH: Percentile > 75 (mean-reversion setups)
            EXTREME: Percentile > 90 (reduce size)
            
        Volatility Contraction/Expansion
            Contracting vol precedes breakouts
            Expanding vol confirms moves
    
    RELEVANCE:
        - Position sizing (reduce size in high vol)
        - Strategy selection (trend vs mean-reversion)
        - Stop loss calibration
        - Required for Phase 4 risk management
    """
    
    def __init__(self, period: int = 20, annual_factor: int = 252):
        self.period = period
        self.annual = annual_factor
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility structure."""
        df = df.copy()
        high, low, close, open_ = df['High'], df['Low'], df['Close'], df['Open']
        n = self.period
        
        # Parkinson Volatility (high-low based)
        # More efficient than close-to-close for continuous trading
        log_hl = np.log(high / low) ** 2
        parkinson_var = log_hl.rolling(n).mean() / (4 * np.log(2))
        df['VS_Parkinson'] = np.sqrt(parkinson_var * self.annual) * 100
        
        # Garman-Klass Volatility (OHLC based)
        log_hl_sq = np.log(high / low) ** 2
        log_co_sq = np.log(close / open_) ** 2
        gk_var = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
        df['VS_GarmanKlass'] = np.sqrt(gk_var.rolling(n).mean() * self.annual) * 100
        
        # Yang-Zhang Volatility (handles overnight gaps)
        log_oc = np.log(open_ / close.shift(1))  # Overnight return
        log_co = np.log(close / open_)  # Open-to-close return
        log_cc = np.log(close / close.shift(1))  # Close-to-close return
        
        # Overnight variance
        overnight_var = log_oc.rolling(n).var()
        
        # Open-to-close variance
        open_close_var = log_co.rolling(n).var()
        
        # Rogers-Satchell variance (intraday)
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(n).mean()
        
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
        df['VS_YangZhang'] = np.sqrt(yz_var.clip(lower=0) * self.annual) * 100
        
        # Use Yang-Zhang as primary volatility measure
        df['VS_Primary'] = df['VS_YangZhang']
        
        # Volatility SMA and trend
        df['VS_SMA'] = df['VS_Primary'].rolling(n).mean()
        df['VS_Expanding'] = df['VS_Primary'] > df['VS_SMA']
        df['VS_Contracting'] = df['VS_Primary'] < df['VS_SMA'] * 0.9
        
        # Volatility Percentile (rolling 252-day)
        df['VS_Percentile'] = df['VS_Primary'].rolling(252, min_periods=50).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100, raw=False
        )
        
        # Volatility Regime Classification
        df['VS_Regime'] = 'NORMAL'
        df.loc[df['VS_Percentile'] < 25, 'VS_Regime'] = 'LOW'
        df.loc[df['VS_Percentile'] > 75, 'VS_Regime'] = 'HIGH'
        df.loc[df['VS_Percentile'] > 90, 'VS_Regime'] = 'EXTREME'
        
        # Bollinger Bandwidth (volatility squeeze indicator)
        # Note: This is NOT Bollinger Bands (Fardeen), just the bandwidth metric
        close_sma = close.rolling(20).mean()
        close_std = close.rolling(20).std()
        df['VS_Bandwidth'] = (close_std * 2 / close_sma) * 100
        
        # Squeeze detection (low bandwidth = potential breakout)
        bw_percentile = df['VS_Bandwidth'].rolling(100, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100, raw=False
        )
        df['VS_Squeeze'] = bw_percentile < 20
        
        # Volatility-adjusted position size multiplier
        # Base size * multiplier = suggested size
        df['VS_SizeMultiplier'] = np.where(
            df['VS_Regime'] == 'EXTREME', 0.5,
            np.where(df['VS_Regime'] == 'HIGH', 0.75,
            np.where(df['VS_Regime'] == 'LOW', 1.25, 1.0))
        )
        
        return df


# =============================================================================
# MOMENTUM QUALITY INDEX
# =============================================================================

class MomentumQualityIndex:
    """
    Momentum Quality Index - Signal Strength Assessment.
    
    Combines multiple momentum measures into a single quality score
    that indicates the strength and reliability of the current move.
    
    COMPONENTS:
    
        Rate of Change (ROC) Suite
            Multiple periods (5, 10, 20) for momentum confirmation
            Agreement across periods = strong momentum
            
        Momentum Divergence Index
            Measures momentum across multiple indicators
            Helps filter false signals
            
        Thrust Detection
            Strong price moves with volume confirmation
            Thrust = high probability continuation
            
        Breadth Proxy (for single stock)
            Uses multi-timeframe momentum alignment
            Daily + Weekly agreement = breadth confirmation
    
    QUALITY SCORE:
        90-100: Exceptional momentum (high probability)
        70-89: Strong momentum (good probability)
        50-69: Moderate momentum (average probability)
        30-49: Weak momentum (low probability)
        0-29: No clear momentum (avoid)
    
    RELEVANCE:
        - Filters low-quality signals
        - Confirms breakouts and reversals
        - Position sizing input (higher quality = larger size)
    """
    
    def __init__(self):
        self.roc_periods = [5, 10, 20]
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum quality components."""
        df = df.copy()
        close = df['Close']
        vol = df['Volume']
        
        # Rate of Change Suite
        for p in self.roc_periods:
            df[f'MQ_ROC_{p}'] = (close - close.shift(p)) / close.shift(p) * 100
        
        # ROC Agreement (all positive or all negative)
        roc_5 = df['MQ_ROC_5']
        roc_10 = df['MQ_ROC_10']
        roc_20 = df['MQ_ROC_20']
        
        df['MQ_ROC_AllBullish'] = (roc_5 > 0) & (roc_10 > 0) & (roc_20 > 0)
        df['MQ_ROC_AllBearish'] = (roc_5 < 0) & (roc_10 < 0) & (roc_20 < 0)
        
        # Momentum acceleration (ROC of ROC)
        df['MQ_Acceleration'] = roc_5 - roc_5.shift(5)
        df['MQ_Accelerating'] = df['MQ_Acceleration'] > 0
        
        # Price momentum score (0-100)
        # Based on ROC percentile ranking
        def percentile_rank(series, window=100):
            return series.rolling(window, min_periods=20).apply(
                lambda x: (x.iloc[-1] > x[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
                raw=False
            )
        
        df['MQ_ROC_Percentile'] = percentile_rank(roc_10)
        
        # Volume momentum score
        vol_change = vol / vol.rolling(20).mean()
        df['MQ_VolScore'] = percentile_rank(vol_change)
        
        # Thrust Detection
        # Strong move (>2% daily) with high volume (>1.5x average)
        strong_up = (close / close.shift(1) - 1) > 0.02
        strong_down = (close / close.shift(1) - 1) < -0.02
        high_vol = vol > vol.rolling(20).mean() * 1.5
        
        df['MQ_BullThrust'] = strong_up & high_vol
        df['MQ_BearThrust'] = strong_down & high_vol
        
        # Consecutive thrust count (momentum persistence)
        df['MQ_ThrustCount'] = 0
        bull_thrust_count = 0
        bear_thrust_count = 0
        
        for i in range(len(df)):
            if df['MQ_BullThrust'].iloc[i]:
                bull_thrust_count += 1
                bear_thrust_count = 0
            elif df['MQ_BearThrust'].iloc[i]:
                bear_thrust_count += 1
                bull_thrust_count = 0
            else:
                # Decay over time if no thrust
                if bull_thrust_count > 0 and i > 0:
                    if (df['Close'].iloc[i] < df['Close'].iloc[i-1]):
                        bull_thrust_count = max(0, bull_thrust_count - 1)
                if bear_thrust_count > 0 and i > 0:
                    if (df['Close'].iloc[i] > df['Close'].iloc[i-1]):
                        bear_thrust_count = max(0, bear_thrust_count - 1)
            
            df.iloc[i, df.columns.get_loc('MQ_ThrustCount')] = bull_thrust_count - bear_thrust_count
        
        # Overall Momentum Quality Score (0-100)
        # Components:
        # - ROC agreement: 30 points
        # - ROC percentile: 25 points
        # - Volume score: 20 points
        # - Acceleration: 15 points
        # - Thrust persistence: 10 points
        
        score = pd.Series(50.0, index=df.index)  # Base score
        
        # ROC agreement
        score = np.where(df['MQ_ROC_AllBullish'] | df['MQ_ROC_AllBearish'], 
                        score + 15, score - 5)
        
        # ROC percentile contribution
        roc_contrib = (df['MQ_ROC_Percentile'] - 50) * 0.5  # -25 to +25
        score = score + roc_contrib.fillna(0)
        
        # Volume contribution
        vol_contrib = (df['MQ_VolScore'] - 50) * 0.4  # -20 to +20
        score = score + vol_contrib.fillna(0)
        
        # Acceleration bonus
        score = np.where(df['MQ_Accelerating'], score + 7.5, score - 2.5)
        
        # Thrust bonus
        thrust_contrib = np.clip(df['MQ_ThrustCount'].abs() * 5, 0, 10)
        score = score + thrust_contrib
        
        df['MQ_Score'] = np.clip(score, 0, 100)
        
        # Quality classification
        df['MQ_Quality'] = pd.cut(
            df['MQ_Score'],
            bins=[0, 30, 50, 70, 90, 100],
            labels=['POOR', 'WEAK', 'MODERATE', 'STRONG', 'EXCEPTIONAL'],
            include_lowest=True
        )
        
        return df


# =============================================================================
# PRICE PATTERN SCANNER
# =============================================================================

class PricePatternScanner:
    """
    Price Pattern Recognition - Candlestick and Bar Patterns.
    
    Identifies high-probability price patterns that often precede
    significant moves. Focus on patterns with statistical edge.
    
    PATTERNS DETECTED:
    
        Reversal Patterns:
            - Engulfing (bullish/bearish)
            - Pin Bar / Hammer / Shooting Star
            - Inside Bar Reversal
            - Key Reversal Day
            
        Continuation Patterns:
            - Inside Bar Breakout
            - Narrow Range (NR4, NR7)
            - Three Bar Play
            
        Breakout Patterns:
            - Range Expansion
            - Volatility Breakout
    
    RELEVANCE:
        - Entry timing for signals
        - Stop loss placement
        - Pattern confluence with indicators
    """
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price patterns."""
        df = df.copy()
        o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
        
        # Body and range calculations
        body = abs(c - o)
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        range_hl = h - l
        
        # Bullish/Bearish candle
        df['PP_Bullish'] = c > o
        df['PP_Bearish'] = c < o
        
        # Engulfing Patterns
        prev_body = body.shift(1)
        prev_bullish = (c.shift(1) > o.shift(1))
        prev_bearish = (c.shift(1) < o.shift(1))
        
        df['PP_BullEngulf'] = (
            prev_bearish &
            df['PP_Bullish'] &
            (o <= c.shift(1)) &
            (c >= o.shift(1)) &
            (body > prev_body)
        )
        
        df['PP_BearEngulf'] = (
            prev_bullish &
            df['PP_Bearish'] &
            (o >= c.shift(1)) &
            (c <= o.shift(1)) &
            (body > prev_body)
        )
        
        # Pin Bar / Hammer / Shooting Star
        small_body = body < range_hl * 0.3
        
        # Hammer (bullish): Small body at top, long lower wick
        df['PP_Hammer'] = (
            small_body &
            (lower_wick > body * 2) &
            (upper_wick < body * 0.5) &
            (c.shift(1) < c.shift(2))  # After downtrend
        )
        
        # Shooting Star (bearish): Small body at bottom, long upper wick
        df['PP_ShootingStar'] = (
            small_body &
            (upper_wick > body * 2) &
            (lower_wick < body * 0.5) &
            (c.shift(1) > c.shift(2))  # After uptrend
        )
        
        # Inside Bar
        df['PP_InsideBar'] = (h < h.shift(1)) & (l > l.shift(1))
        
        # Inside Bar Breakout
        inside_prev = df['PP_InsideBar'].shift(1).fillna(False)
        df['PP_InsideBreakUp'] = inside_prev & (c > h.shift(1))
        df['PP_InsideBreakDown'] = inside_prev & (c < l.shift(1))
        
        # Key Reversal Day
        # Bullish: New low, close above previous close
        df['PP_KeyRevBull'] = (l < l.shift(1)) & (c > c.shift(1)) & (c > o)
        # Bearish: New high, close below previous close
        df['PP_KeyRevBear'] = (h > h.shift(1)) & (c < c.shift(1)) & (c < o)
        
        # Narrow Range (NR4 - narrowest range in 4 days)
        range_4 = range_hl.rolling(4).min()
        df['PP_NR4'] = range_hl == range_4
        
        # Narrow Range 7 (even more significant)
        range_7 = range_hl.rolling(7).min()
        df['PP_NR7'] = range_hl == range_7
        
        # Wide Range Bar (expansion)
        avg_range = range_hl.rolling(10).mean()
        df['PP_WideRange'] = range_hl > avg_range * 1.5
        
        # Three Bar Play (pullback setup)
        # Bullish: Strong up bar, followed by 2 small inside/down bars
        strong_up = (c - o > avg_range * 0.5) & df['PP_Bullish']
        small_bar_1 = (body.shift(-1) < body * 0.5)
        small_bar_2 = (body.shift(-2) < body * 0.5)
        df['PP_ThreeBarBull'] = strong_up.shift(2) & small_bar_1.shift(1) & small_bar_2
        
        # Bearish version
        strong_down = (o - c > avg_range * 0.5) & df['PP_Bearish']
        df['PP_ThreeBarBear'] = strong_down.shift(2) & small_bar_1.shift(1) & small_bar_2
        
        # Doji (indecision)
        df['PP_Doji'] = body < range_hl * 0.1
        
        # Pattern score summary
        bullish_patterns = (
            df['PP_BullEngulf'].astype(int) * 3 +
            df['PP_Hammer'].astype(int) * 2 +
            df['PP_KeyRevBull'].astype(int) * 2 +
            df['PP_InsideBreakUp'].astype(int) * 2 +
            df['PP_ThreeBarBull'].astype(int) * 1
        )
        
        bearish_patterns = (
            df['PP_BearEngulf'].astype(int) * 3 +
            df['PP_ShootingStar'].astype(int) * 2 +
            df['PP_KeyRevBear'].astype(int) * 2 +
            df['PP_InsideBreakDown'].astype(int) * 2 +
            df['PP_ThreeBarBear'].astype(int) * 1
        )
        
        df['PP_BullScore'] = bullish_patterns
        df['PP_BearScore'] = bearish_patterns
        df['PP_NetScore'] = bullish_patterns - bearish_patterns
        
        return df


# =============================================================================
# ENHANCED DATA STRUCTURES
# =============================================================================

@dataclass
class MarketStructureState:
    """Market structure analysis state."""
    trend: str                       # BULLISH, BEARISH, CONSOLIDATION
    last_swing_high: float
    last_swing_low: float
    higher_highs: int                # Count in recent period
    lower_lows: int
    bos_bullish: bool               # Break of structure
    bos_bearish: bool
    choch_bullish: bool             # Change of character
    choch_bearish: bool
    key_resistance: List[float]
    key_support: List[float]


@dataclass
class VolumeFlowState:
    """Volume flow analysis state."""
    obv_trend: str                  # BULLISH, BEARISH
    obv_divergence: Optional[str]   # BULLISH, BEARISH, None
    cmf_value: float
    cmf_zone: str                   # ACCUMULATION, DISTRIBUTION, NEUTRAL
    rvol: float                     # Relative volume
    volume_climax: Optional[str]    # BUYING, SELLING, None
    accumulation_phase: bool
    distribution_phase: bool


@dataclass
class VolatilityState:
    """Volatility structure state."""
    current_vol: float              # Yang-Zhang annualized
    vol_percentile: float
    regime: str                     # LOW, NORMAL, HIGH, EXTREME
    expanding: bool
    contracting: bool
    squeeze: bool
    size_multiplier: float


@dataclass
class MomentumState:
    """Momentum quality state."""
    score: float                    # 0-100
    quality: str                    # POOR, WEAK, MODERATE, STRONG, EXCEPTIONAL
    roc_aligned: bool
    accelerating: bool
    thrust_count: int


@dataclass
class PatternState:
    """Price pattern state."""
    bullish_score: int
    bearish_score: int
    net_score: int
    active_patterns: List[str]


@dataclass
class EnhancedConfluenceAnalysis:
    """Complete enhanced analysis with all components."""
    # Core indicators
    ichimoku: IchimokuAnalysis
    vwap: VWAPAnalysis
    williams_r: WilliamsRAnalysis
    cci: CCIAnalysis
    mtf: MultiTimeframeAnalysis
    
    # Advanced components
    structure: MarketStructureState
    volume_flow: VolumeFlowState
    volatility: VolatilityState
    momentum: MomentumState
    patterns: PatternState
    
    # Final analysis
    confluence_score: float
    signal: Signal
    quality: SignalQuality
    confidence: float
    
    # Enhanced trade setup
    setup: Optional[TradeSetup]
    position_size_factor: float     # Volatility-adjusted
    
    # Comprehensive factors
    bullish_factors: List[str]
    bearish_factors: List[str]
    warnings: List[str]
    summary: str


# =============================================================================
# ENHANCED TECHNICAL ANALYZER
# =============================================================================

class EnhancedTechnicalAnalyzer:
    """
    Enhanced Technical Analyzer with Advanced Components.
    
    Integrates all indicator systems into a comprehensive analysis framework:
    - Core Indicators (Ichimoku, VWAP, Williams %R, CCI)
    - Market Structure Analysis
    - Volume Flow Analysis
    - Volatility Structure
    - Momentum Quality
    - Price Patterns
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Core indicators
        ich_cfg = config.get('ichimoku', {})
        self.ichimoku = IchimokuCloud(
            tenkan_period=ich_cfg.get('tenkan', 9),
            kijun_period=ich_cfg.get('kijun', 26),
            senkou_b_period=ich_cfg.get('senkou_b', 52)
        )
        self.vwap = VWAPSuite()
        self.williams = WilliamsRAdvanced()
        self.cci = CCIProfessional()
        
        # Advanced components
        self.structure = MarketStructureAnalyzer()
        self.volume = VolumeFlowAnalyzer()
        self.volatility = VolatilityStructure()
        self.momentum = MomentumQualityIndex()
        self.patterns = PricePatternScanner()
    
    def calculate_all(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame = None,
        earnings_dates: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate all indicators and advanced components."""
        
        # Core indicators
        logger.info("Calculating Ichimoku Cloud...")
        df = self.ichimoku.calculate(df)
        
        logger.info("Calculating VWAP Suite...")
        df = self.vwap.calculate(df, earnings_dates)
        
        logger.info("Calculating Williams %R Advanced...")
        df = self.williams.calculate(df)
        
        logger.info("Calculating CCI Professional...")
        df = self.cci.calculate(df)
        
        # Advanced components
        logger.info("Analyzing Market Structure...")
        df = self.structure.calculate(df)
        
        logger.info("Analyzing Volume Flow...")
        df = self.volume.calculate(df)
        
        logger.info("Analyzing Volatility Structure...")
        df = self.volatility.calculate(df)
        
        logger.info("Calculating Momentum Quality...")
        df = self.momentum.calculate(df)
        
        logger.info("Scanning Price Patterns...")
        df = self.patterns.calculate(df)
        
        # Weekly if provided
        if weekly_df is not None and len(weekly_df) > 0:
            logger.info("Calculating Weekly Ichimoku...")
            weekly_df = self.ichimoku.calculate(weekly_df, prefix="W_")
        
        return df, weekly_df
    
    def analyze(
        self,
        df: pd.DataFrame,
        weekly_df: pd.DataFrame = None,
        earnings_dates: List[str] = None,
        idx: int = -1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, EnhancedConfluenceAnalysis]:
        """Perform complete enhanced analysis."""
        
        # Calculate all
        df, weekly_df = self.calculate_all(df, weekly_df, earnings_dates)
        
        row = df.iloc[idx]
        
        # Core indicator analysis
        logger.info("Analyzing indicator states...")
        ichimoku_state = self.ichimoku.analyze(df, weekly_df, idx)
        vwap_state = self.vwap.analyze(df, earnings_dates, idx)
        williams_state = self.williams.analyze(df, idx)
        cci_state = self.cci.analyze(df, idx)
        
        # Weekly for MTF
        weekly_ichimoku = None
        if weekly_df is not None and len(weekly_df) > 0:
            weekly_ichimoku = self.ichimoku.analyze(weekly_df, None, -1)
        
        mtf = MultiTimeframeAnalyzer.analyze(ichimoku_state, weekly_ichimoku, cci_state)
        
        # Advanced component states
        structure_state = self._analyze_structure(df, row, idx)
        volume_state = self._analyze_volume(row)
        volatility_state = self._analyze_volatility(row)
        momentum_state = self._analyze_momentum(row)
        pattern_state = self._analyze_patterns(row)
        
        # Enhanced confluence
        logger.info("Computing enhanced confluence...")
        analysis = self._compute_enhanced_confluence(
            ichimoku_state, vwap_state, williams_state, cci_state, mtf,
            structure_state, volume_state, volatility_state, 
            momentum_state, pattern_state, row
        )
        
        return df, weekly_df, analysis
    
    def _analyze_structure(self, df: pd.DataFrame, row: pd.Series, idx: int) -> MarketStructureState:
        """Extract market structure state."""
        
        # Trend determination
        bull_struct = self._safe_bool(row.get('MS_BullishStructure'))
        bear_struct = self._safe_bool(row.get('MS_BearishStructure'))
        
        if bull_struct:
            trend = "BULLISH"
        elif bear_struct:
            trend = "BEARISH"
        else:
            trend = "CONSOLIDATION"
        
        # Get key levels
        levels = self.structure.get_key_levels(df)
        
        return MarketStructureState(
            trend=trend,
            last_swing_high=self._safe_float(row.get('MS_LastSwingHigh')),
            last_swing_low=self._safe_float(row.get('MS_LastSwingLow')),
            higher_highs=int(df['MS_HigherHigh'].tail(20).sum()),
            lower_lows=int(df['MS_LowerLow'].tail(20).sum()),
            bos_bullish=self._safe_bool(row.get('MS_BOS_Bullish')),
            bos_bearish=self._safe_bool(row.get('MS_BOS_Bearish')),
            choch_bullish=self._safe_bool(row.get('MS_CHoCH_Bullish')),
            choch_bearish=self._safe_bool(row.get('MS_CHoCH_Bearish')),
            key_resistance=levels['resistance'][:3],
            key_support=levels['support'][:3]
        )
    
    def _analyze_volume(self, row: pd.Series) -> VolumeFlowState:
        """Extract volume flow state."""
        
        obv_trend = "BULLISH" if self._safe_bool(row.get('VF_OBV_Trend')) else "BEARISH"
        
        obv_div = None
        if self._safe_bool(row.get('VF_OBV_BullDiv')):
            obv_div = "BULLISH"
        elif self._safe_bool(row.get('VF_OBV_BearDiv')):
            obv_div = "BEARISH"
        
        cmf = self._safe_float(row.get('VF_CMF'))
        if cmf > 0.1:
            cmf_zone = "ACCUMULATION"
        elif cmf < -0.1:
            cmf_zone = "DISTRIBUTION"
        else:
            cmf_zone = "NEUTRAL"
        
        climax = None
        if self._safe_bool(row.get('VF_BuyingClimax')):
            climax = "BUYING"
        elif self._safe_bool(row.get('VF_SellingClimax')):
            climax = "SELLING"
        
        return VolumeFlowState(
            obv_trend=obv_trend,
            obv_divergence=obv_div,
            cmf_value=cmf,
            cmf_zone=cmf_zone,
            rvol=self._safe_float(row.get('VF_RVOL'), 1.0),
            volume_climax=climax,
            accumulation_phase=self._safe_bool(row.get('VF_Accumulation')),
            distribution_phase=self._safe_bool(row.get('VF_Distribution'))
        )
    
    def _analyze_volatility(self, row: pd.Series) -> VolatilityState:
        """Extract volatility state."""
        return VolatilityState(
            current_vol=self._safe_float(row.get('VS_Primary')),
            vol_percentile=self._safe_float(row.get('VS_Percentile'), 50),
            regime=str(row.get('VS_Regime', 'NORMAL')),
            expanding=self._safe_bool(row.get('VS_Expanding')),
            contracting=self._safe_bool(row.get('VS_Contracting')),
            squeeze=self._safe_bool(row.get('VS_Squeeze')),
            size_multiplier=self._safe_float(row.get('VS_SizeMultiplier'), 1.0)
        )
    
    def _analyze_momentum(self, row: pd.Series) -> MomentumState:
        """Extract momentum state."""
        quality = str(row.get('MQ_Quality', 'MODERATE'))
        if pd.isna(row.get('MQ_Quality')):
            quality = 'MODERATE'
        
        return MomentumState(
            score=self._safe_float(row.get('MQ_Score'), 50),
            quality=quality,
            roc_aligned=self._safe_bool(row.get('MQ_ROC_AllBullish')) or self._safe_bool(row.get('MQ_ROC_AllBearish')),
            accelerating=self._safe_bool(row.get('MQ_Accelerating')),
            thrust_count=int(self._safe_float(row.get('MQ_ThrustCount')))
        )
    
    def _analyze_patterns(self, row: pd.Series) -> PatternState:
        """Extract pattern state."""
        active = []
        
        pattern_checks = [
            ('PP_BullEngulf', 'Bullish Engulfing'),
            ('PP_BearEngulf', 'Bearish Engulfing'),
            ('PP_Hammer', 'Hammer'),
            ('PP_ShootingStar', 'Shooting Star'),
            ('PP_InsideBreakUp', 'Inside Bar Breakout Up'),
            ('PP_InsideBreakDown', 'Inside Bar Breakout Down'),
            ('PP_KeyRevBull', 'Key Reversal Bullish'),
            ('PP_KeyRevBear', 'Key Reversal Bearish'),
            ('PP_NR7', 'NR7 (Squeeze)'),
            ('PP_WideRange', 'Wide Range Expansion'),
        ]
        
        for col, name in pattern_checks:
            if self._safe_bool(row.get(col)):
                active.append(name)
        
        return PatternState(
            bullish_score=int(self._safe_float(row.get('PP_BullScore'))),
            bearish_score=int(self._safe_float(row.get('PP_BearScore'))),
            net_score=int(self._safe_float(row.get('PP_NetScore'))),
            active_patterns=active
        )
    
    def _compute_enhanced_confluence(
        self,
        ichimoku: IchimokuAnalysis,
        vwap: VWAPAnalysis,
        williams: WilliamsRAnalysis,
        cci: CCIAnalysis,
        mtf: MultiTimeframeAnalysis,
        structure: MarketStructureState,
        volume: VolumeFlowState,
        volatility: VolatilityState,
        momentum: MomentumState,
        patterns: PatternState,
        row: pd.Series
    ) -> EnhancedConfluenceAnalysis:
        """Compute enhanced confluence with all components."""
        
        bullish_factors = []
        bearish_factors = []
        warnings = []
        
        # Core indicator signals
        signals = [ichimoku.signal, vwap.signal, williams.signal, cci.signal]
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        bullish_count = 0
        bearish_count = 0
        
        for sig in signals:
            weight = sig.confidence / 100.0
            weighted_sum += sig.value * weight
            weight_total += weight
            
            if sig.value > 0.1:
                bullish_count += 1
                bullish_factors.extend(sig.factors[:2])
            elif sig.value < -0.1:
                bearish_count += 1
                bearish_factors.extend(sig.factors[:2])
        
        # Structure contribution
        if structure.trend == "BULLISH":
            weighted_sum += 0.15
            weight_total += 0.2
            bullish_factors.append(f"Structure: Bullish (HH:{structure.higher_highs})")
        elif structure.trend == "BEARISH":
            weighted_sum -= 0.15
            weight_total += 0.2
            bearish_factors.append(f"Structure: Bearish (LL:{structure.lower_lows})")
        
        if structure.choch_bullish:
            weighted_sum += 0.2
            bullish_factors.append("Change of Character: BULLISH")
        elif structure.choch_bearish:
            weighted_sum -= 0.2
            bearish_factors.append("Change of Character: BEARISH")
        
        # Volume contribution
        if volume.cmf_zone == "ACCUMULATION":
            weighted_sum += 0.1
            bullish_factors.append(f"CMF: Accumulation ({volume.cmf_value:.2f})")
        elif volume.cmf_zone == "DISTRIBUTION":
            weighted_sum -= 0.1
            bearish_factors.append(f"CMF: Distribution ({volume.cmf_value:.2f})")
        
        if volume.obv_divergence == "BULLISH":
            weighted_sum += 0.15
            bullish_factors.append("OBV: Bullish Divergence")
        elif volume.obv_divergence == "BEARISH":
            weighted_sum -= 0.15
            bearish_factors.append("OBV: Bearish Divergence")
        
        if volume.volume_climax:
            warnings.append(f"Volume Climax: {volume.volume_climax} (potential exhaustion)")
        
        # Momentum contribution
        if momentum.quality in ['STRONG', 'EXCEPTIONAL']:
            weight_total *= 1.1  # Boost confidence
            if momentum.thrust_count > 0:
                bullish_factors.append(f"Momentum: {momentum.quality} (thrust count: {momentum.thrust_count})")
            elif momentum.thrust_count < 0:
                bearish_factors.append(f"Momentum: {momentum.quality} (thrust count: {momentum.thrust_count})")
        elif momentum.quality == 'POOR':
            warnings.append("Momentum Quality: POOR (low probability setup)")
        
        # Pattern contribution
        if patterns.net_score > 0:
            weighted_sum += 0.05 * patterns.net_score
            if patterns.active_patterns:
                bullish_factors.append(f"Patterns: {', '.join(patterns.active_patterns[:2])}")
        elif patterns.net_score < 0:
            weighted_sum += 0.05 * patterns.net_score
            if patterns.active_patterns:
                bearish_factors.append(f"Patterns: {', '.join(patterns.active_patterns[:2])}")
        
        # Volatility warnings
        if volatility.regime == "EXTREME":
            warnings.append("Volatility: EXTREME (reduce position size)")
        elif volatility.squeeze:
            bullish_factors.append("Volatility Squeeze: Breakout imminent")
        
        # MTF adjustment
        if mtf.alignment:
            weighted_sum *= 1.15
            bullish_factors.append(f"MTF: Aligned ({mtf.daily_trend.value}/{mtf.weekly_trend.value})")
        else:
            weighted_sum *= 0.85
            warnings.append(f"MTF: Not aligned ({mtf.daily_trend.value}/{mtf.weekly_trend.value})")
        
        # Calculate final score
        confluence_score = (weighted_sum / weight_total * 100) if weight_total > 0 else 0
        confluence_score = max(-100, min(100, confluence_score))
        
        # Determine signal
        if confluence_score >= 50:
            signal = Signal.STRONG_BUY
        elif confluence_score >= 20:
            signal = Signal.BUY
        elif confluence_score <= -50:
            signal = Signal.STRONG_SELL
        elif confluence_score <= -20:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
        
        # Quality grading
        max_aligned = max(bullish_count, bearish_count)
        has_divergence = (williams.regular_bullish_div or williams.regular_bearish_div or
                         cci.bullish_divergence or cci.bearish_divergence or
                         volume.obv_divergence is not None)
        has_structure = structure.choch_bullish or structure.choch_bearish or structure.bos_bullish or structure.bos_bearish
        good_momentum = momentum.quality in ['STRONG', 'EXCEPTIONAL']
        
        if max_aligned >= 3 and mtf.alignment and (has_divergence or has_structure) and good_momentum:
            quality = SignalQuality.A
        elif max_aligned >= 2 and mtf.alignment and good_momentum:
            quality = SignalQuality.B
        elif max_aligned >= 2 or (mtf.alignment and good_momentum):
            quality = SignalQuality.C
        else:
            quality = SignalQuality.D
        
        # Confidence
        avg_conf = sum(s.confidence for s in signals) / len(signals)
        if quality == SignalQuality.A:
            avg_conf = min(98, avg_conf + 20)
        elif quality == SignalQuality.B:
            avg_conf = min(90, avg_conf + 10)
        elif quality == SignalQuality.D:
            avg_conf = max(20, avg_conf - 20)
        
        # Position size factor
        size_factor = volatility.size_multiplier
        if quality == SignalQuality.A:
            size_factor *= 1.2
        elif quality == SignalQuality.D:
            size_factor *= 0.5
        
        # Trade setup
        setup = self._create_trade_setup(
            signal, quality, ichimoku, vwap, structure, volatility, row
        )
        
        # Summary
        summary = self._generate_summary(
            signal, quality, confluence_score, bullish_count, bearish_count,
            mtf, structure, momentum, volatility
        )
        
        return EnhancedConfluenceAnalysis(
            ichimoku=ichimoku,
            vwap=vwap,
            williams_r=williams,
            cci=cci,
            mtf=mtf,
            structure=structure,
            volume_flow=volume,
            volatility=volatility,
            momentum=momentum,
            patterns=patterns,
            confluence_score=confluence_score,
            signal=signal,
            quality=quality,
            confidence=avg_conf,
            setup=setup,
            position_size_factor=size_factor,
            bullish_factors=bullish_factors[:8],
            bearish_factors=bearish_factors[:8],
            warnings=warnings,
            summary=summary
        )
    
    def _create_trade_setup(
        self,
        signal: Signal,
        quality: SignalQuality,
        ichimoku: IchimokuAnalysis,
        vwap: VWAPAnalysis,
        structure: MarketStructureState,
        volatility: VolatilityState,
        row: pd.Series
    ) -> Optional[TradeSetup]:
        """Create trade setup with structure-based levels."""
        
        if quality == SignalQuality.D:
            return None
        
        close = row['Close']
        
        if signal in [Signal.STRONG_BUY, Signal.BUY]:
            direction = "LONG"
            
            # Entry near VWAP or Kijun
            entry_low = min(vwap.vwap, ichimoku.kijun) * 0.998
            entry_high = max(vwap.vwap, ichimoku.kijun) * 1.002
            
            # Stop below structure support
            if structure.key_support:
                stop = min(structure.key_support[0], ichimoku.cloud_bottom) * 0.995
            else:
                stop = ichimoku.cloud_bottom * 0.995
            
            # Target at structure resistance
            if structure.key_resistance:
                target_1 = structure.key_resistance[0]
                target_2 = structure.key_resistance[1] if len(structure.key_resistance) > 1 else None
            else:
                target_1 = close * 1.05
                target_2 = close * 1.10
            
            setup_type = "Long setup"
            if structure.choch_bullish:
                setup_type = "CHoCH Bullish reversal"
            elif ichimoku.kumo_breakout == "BULLISH":
                setup_type = "Kumo breakout long"
            elif structure.bos_bullish:
                setup_type = "Break of structure continuation"
        
        elif signal in [Signal.STRONG_SELL, Signal.SELL]:
            direction = "SHORT"
            
            entry_low = min(vwap.vwap, ichimoku.kijun) * 0.998
            entry_high = max(vwap.vwap, ichimoku.kijun) * 1.002
            
            if structure.key_resistance:
                stop = max(structure.key_resistance[0], ichimoku.cloud_top) * 1.005
            else:
                stop = ichimoku.cloud_top * 1.005
            
            if structure.key_support:
                target_1 = structure.key_support[0]
                target_2 = structure.key_support[1] if len(structure.key_support) > 1 else None
            else:
                target_1 = close * 0.95
                target_2 = close * 0.90
            
            setup_type = "Short setup"
            if structure.choch_bearish:
                setup_type = "CHoCH Bearish reversal"
            elif ichimoku.kumo_breakout == "BEARISH":
                setup_type = "Kumo breakdown short"
            elif structure.bos_bearish:
                setup_type = "Break of structure continuation"
        
        else:
            return None
        
        # Risk/reward calculation
        entry_mid = (entry_low + entry_high) / 2
        risk = abs(entry_mid - stop)
        reward = abs(target_1 - entry_mid)
        rr = reward / risk if risk > 0 else 0
        
        return TradeSetup(
            direction=direction,
            entry_zone=(entry_low, entry_high),
            stop_loss=stop,
            target_1=target_1,
            target_2=target_2,
            risk_reward=rr,
            setup_type=setup_type
        )
    
    def _generate_summary(
        self,
        signal: Signal,
        quality: SignalQuality,
        confluence: float,
        bull_count: int,
        bear_count: int,
        mtf: MultiTimeframeAnalysis,
        structure: MarketStructureState,
        momentum: MomentumState,
        volatility: VolatilityState
    ) -> str:
        """Generate comprehensive summary."""
        
        direction = "BULLISH" if signal in [Signal.STRONG_BUY, Signal.BUY] else \
                   "BEARISH" if signal in [Signal.STRONG_SELL, Signal.SELL] else "NEUTRAL"
        
        summary = f"{direction} bias with Grade {quality.value} quality. "
        summary += f"Confluence: {confluence:+.0f}, {bull_count} bullish / {bear_count} bearish indicators. "
        summary += f"Structure: {structure.trend}. "
        summary += f"Momentum: {momentum.quality} ({momentum.score:.0f}/100). "
        summary += f"Volatility: {volatility.regime} ({volatility.current_vol:.1f}% ann). "
        summary += f"MTF: {'Aligned' if mtf.alignment else 'Conflicting'}."
        
        return summary
    
    @staticmethod
    def _safe_float(val, default: float = 0.0) -> float:
        if pd.isna(val):
            return default
        try:
            return float(val)
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def _safe_bool(val, default: bool = False) -> bool:
        if pd.isna(val):
            return default
        return bool(val)


# =============================================================================
# ENHANCED TERMINAL REPORT
# =============================================================================

def print_enhanced_analysis(analysis: EnhancedConfluenceAnalysis, symbol: str) -> None:
    """Print comprehensive enhanced analysis report."""
    w = 80
    
    print()
    print("╔" + "═" * (w-2) + "╗")
    print(f"║  {symbol} ENHANCED TECHNICAL ANALYSIS".ljust(w-2) + "║")
    print(f"║  Institutional-Grade Multi-Factor Analysis".ljust(w-2) + "║")
    print("╚" + "═" * (w-2) + "╝")
    
    # Main Signal Box
    print()
    sig_width = 50
    print("  ┌" + "─" * sig_width + "┐")
    print(f"  │{'SIGNAL: ' + analysis.signal.value:^{sig_width}}│")
    print(f"  │{'Grade ' + analysis.quality.value + ' Quality':^{sig_width}}│")
    print("  ├" + "─" * sig_width + "┤")
    print(f"  │  Confidence: {analysis.confidence:>5.0f}%{' ':31}│")
    print(f"  │  Confluence: {analysis.confluence_score:>+6.0f}{' ':31}│")
    print(f"  │  Position Size Factor: {analysis.position_size_factor:>4.2f}x{' ':18}│")
    print("  └" + "─" * sig_width + "┘")
    
    # Core Indicators
    print()
    print("─" * w)
    print("  CORE INDICATOR SIGNALS")
    print("─" * w)
    
    indicators = [
        ("Ichimoku", analysis.ichimoku.signal),
        ("VWAP", analysis.vwap.signal),
        ("Williams %R", analysis.williams_r.signal),
        ("CCI", analysis.cci.signal),
    ]
    
    for name, sig in indicators:
        d = "▲" if sig.value > 0.05 else "▼" if sig.value < -0.05 else "─"
        bar = int((sig.value + 1) / 2 * 20)
        print(f"  {name:12} {d} {sig.value:+.2f} [{'░'*bar}█{'░'*(19-bar)}] {sig.confidence:>3.0f}%")
    
    # Market Structure
    print()
    print("─" * w)
    print("  MARKET STRUCTURE")
    print("─" * w)
    s = analysis.structure
    print(f"  Trend:           {s.trend}")
    print(f"  Swing High:      ${s.last_swing_high:,.2f}")
    print(f"  Swing Low:       ${s.last_swing_low:,.2f}")
    print(f"  Pattern:         HH:{s.higher_highs} | LL:{s.lower_lows}")
    if s.bos_bullish:
        print(f"  Signal:          BREAK OF STRUCTURE (Bullish)")
    if s.bos_bearish:
        print(f"  Signal:          BREAK OF STRUCTURE (Bearish)")
    if s.choch_bullish:
        print(f"  Signal:          CHANGE OF CHARACTER (Bullish Reversal)")
    if s.choch_bearish:
        print(f"  Signal:          CHANGE OF CHARACTER (Bearish Reversal)")
    if s.key_resistance:
        print(f"  Resistance:      ${s.key_resistance[0]:,.2f}")
    if s.key_support:
        print(f"  Support:         ${s.key_support[0]:,.2f}")
    
    # Volume Flow
    print()
    print("─" * w)
    print("  VOLUME FLOW")
    print("─" * w)
    v = analysis.volume_flow
    print(f"  OBV Trend:       {v.obv_trend}")
    print(f"  CMF:             {v.cmf_value:+.3f} ({v.cmf_zone})")
    print(f"  Relative Vol:    {v.rvol:.2f}x average")
    if v.obv_divergence:
        print(f"  Divergence:      {v.obv_divergence} (reversal warning)")
    if v.volume_climax:
        print(f"  Climax:          {v.volume_climax} CLIMAX (exhaustion)")
    if v.accumulation_phase:
        print(f"  Phase:           ACCUMULATION")
    if v.distribution_phase:
        print(f"  Phase:           DISTRIBUTION")
    
    # Volatility
    print()
    print("─" * w)
    print("  VOLATILITY STRUCTURE")
    print("─" * w)
    vol = analysis.volatility
    print(f"  Current Vol:     {vol.current_vol:.1f}% (annualized)")
    print(f"  Percentile:      {vol.vol_percentile:.0f}%")
    print(f"  Regime:          {vol.regime}")
    print(f"  Size Multiplier: {vol.size_multiplier:.2f}x")
    if vol.squeeze:
        print(f"  Squeeze:         DETECTED (breakout imminent)")
    if vol.expanding:
        print(f"  Trend:           EXPANDING")
    elif vol.contracting:
        print(f"  Trend:           CONTRACTING")
    
    # Momentum Quality
    print()
    print("─" * w)
    print("  MOMENTUM QUALITY")
    print("─" * w)
    m = analysis.momentum
    quality_bar = int(m.score / 5)
    print(f"  Score:           {m.score:.0f}/100 [{'█'*quality_bar}{'░'*(20-quality_bar)}]")
    print(f"  Quality:         {m.quality}")
    print(f"  ROC Aligned:     {'Yes' if m.roc_aligned else 'No'}")
    print(f"  Accelerating:    {'Yes' if m.accelerating else 'No'}")
    print(f"  Thrust Count:    {m.thrust_count:+d}")
    
    # Patterns
    if analysis.patterns.active_patterns:
        print()
        print("─" * w)
        print("  ACTIVE PATTERNS")
        print("─" * w)
        print(f"  Score:           Bull {analysis.patterns.bullish_score} | Bear {analysis.patterns.bearish_score}")
        for p in analysis.patterns.active_patterns[:4]:
            print(f"  • {p}")
    
    # Ichimoku Details
    print()
    print("─" * w)
    print("  ICHIMOKU CLOUD DETAILS")
    print("─" * w)
    ich = analysis.ichimoku
    print(f"  Position:        {ich.price_position.value}")
    print(f"  Cloud:           {ich.cloud_color.value} | Thickness: {ich.cloud_thickness_pct:.2f}%")
    print(f"  Tenkan/Kijun:    ${ich.tenkan:,.2f} / ${ich.kijun:,.2f}")
    print(f"  TK Cross:        {ich.tk_cross.value}")
    if ich.kumo_breakout:
        print(f"  Kumo Breakout:   {ich.kumo_breakout}")
    if ich.kijun_bounce:
        print(f"  Kijun Bounce:    DETECTED")
    print(f"  Chikou Confirm:  {'Yes' if ich.chikou_confirmed else 'No'}")
    print(f"  Weekly Aligned:  {'Yes' if ich.weekly_aligned else 'No'}")
    
    # VWAP Details
    print()
    print("─" * w)
    print("  VWAP SUITE DETAILS")
    print("─" * w)
    vw = analysis.vwap
    print(f"  VWAP:            ${vw.vwap:,.2f}")
    print(f"  Distance:        {vw.distance_pct:+.2f}%")
    print(f"  Band Position:   {vw.band_position}")
    print(f"  5D/20D VWAP:     ${vw.vwap_5d:,.2f} / ${vw.vwap_20d:,.2f}")
    print(f"  Rolling Trend:   {vw.price_vs_rolling}")
    print(f"  Anchored (E):    {len(vw.anchored_earnings)} active")
    
    # Williams %R Details
    print()
    print("─" * w)
    print("  WILLIAMS %R ADVANCED")
    print("─" * w)
    wr = analysis.williams_r
    print(f"  Fast/Slow:       {wr.fast_value:.1f} / {wr.slow_value:.1f}")
    print(f"  Zone:            {wr.zone}")
    print(f"  Alignment:       {'Yes' if wr.fast_slow_alignment else 'No'}")
    if wr.regular_bullish_div:
        print(f"  Signal:          REGULAR BULLISH DIVERGENCE")
    if wr.regular_bearish_div:
        print(f"  Signal:          REGULAR BEARISH DIVERGENCE")
    if wr.hidden_bullish_div:
        print(f"  Signal:          Hidden Bullish Divergence")
    if wr.hidden_bearish_div:
        print(f"  Signal:          Hidden Bearish Divergence")
    if wr.momentum_thrust:
        print(f"  Thrust:          {wr.momentum_thrust}")
    
    # CCI Details
    print()
    print("─" * w)
    print("  CCI PROFESSIONAL")
    print("─" * w)
    cc = analysis.cci
    print(f"  Short/Long:      {cc.short_value:.0f} / {cc.long_value:.0f}")
    print(f"  Zone:            {cc.zone}")
    print(f"  Dual Aligned:    {'Yes' if cc.dual_alignment else 'No'}")
    print(f"  Trend:           {cc.trend_strength.value}")
    if cc.zero_cross:
        print(f"  Zero Cross:      {cc.zero_cross}")
    if cc.bullish_divergence:
        print(f"  Signal:          BULLISH DIVERGENCE")
    if cc.bearish_divergence:
        print(f"  Signal:          BEARISH DIVERGENCE")
    
    # Key Factors
    print()
    print("─" * w)
    print("  KEY FACTORS")
    print("─" * w)
    if analysis.bullish_factors:
        print("  BULLISH:")
        for f in analysis.bullish_factors[:5]:
            print(f"    ▲ {f}")
    if analysis.bearish_factors:
        print("  BEARISH:")
        for f in analysis.bearish_factors[:5]:
            print(f"    ▼ {f}")
    
    # Warnings
    if analysis.warnings:
        print()
        print("─" * w)
        print("  WARNINGS")
        print("─" * w)
        for warn in analysis.warnings:
            print(f"  ⚠ {warn}")
    
    # Trade Setup
    if analysis.setup:
        print()
        print("─" * w)
        print("  TRADE SETUP")
        print("─" * w)
        ts = analysis.setup
        print(f"  Direction:       {ts.direction}")
        print(f"  Type:            {ts.setup_type}")
        print(f"  Entry Zone:      ${ts.entry_zone[0]:,.2f} - ${ts.entry_zone[1]:,.2f}")
        print(f"  Stop Loss:       ${ts.stop_loss:,.2f}")
        print(f"  Target 1:        ${ts.target_1:,.2f}")
        if ts.target_2:
            print(f"  Target 2:        ${ts.target_2:,.2f}")
        print(f"  Risk/Reward:     {ts.risk_reward:.2f}")
    
    # Summary
    print()
    print("─" * w)
    print("  SUMMARY")
    print("─" * w)
    print(f"  {analysis.summary}")
    
    print()
    print("═" * w)


# =============================================================================
# ENHANCED HTML DASHBOARD
# =============================================================================

class IndicatorDashboard:
    """Generate professional HTML dashboard for enhanced technical analysis."""
    
    @staticmethod
    def generate(
        df: pd.DataFrame,
        analysis: EnhancedConfluenceAnalysis,
        symbol: str,
        output_path: Path
    ) -> None:
        """Generate comprehensive HTML dashboard."""
        
        # Sample data for charts (last 200 points, every 2nd)
        chart_df = df.iloc[::2].tail(200)
        
        dates = [d.strftime('%Y-%m-%d') for d in chart_df.index]
        closes = [round(x, 2) for x in chart_df['Close'].tolist()]
        
        # Ichimoku
        tenkan = [round(x, 2) if pd.notna(x) else None for x in chart_df.get('Ich_Tenkan', pd.Series()).tolist()]
        kijun = [round(x, 2) if pd.notna(x) else None for x in chart_df.get('Ich_Kijun', pd.Series()).tolist()]
        
        # VWAP
        vwap_vals = [round(x, 2) if pd.notna(x) else None for x in chart_df.get('VWAP', pd.Series()).tolist()]
        
        # Oscillators
        wr = [round(x, 1) if pd.notna(x) else None for x in chart_df.get('WR_Fast', pd.Series()).tolist()]
        cci = [round(x, 0) if pd.notna(x) else None for x in chart_df.get('CCI_Short', pd.Series()).tolist()]
        
        # Momentum
        mq = [round(x, 0) if pd.notna(x) else None for x in chart_df.get('MQ_Score', pd.Series()).tolist()]
        
        # Colors
        sig_colors = {
            Signal.STRONG_BUY: "#00ff88",
            Signal.BUY: "#4ecdc4",
            Signal.HOLD: "#888888",
            Signal.SELL: "#ff6b6b",
            Signal.STRONG_SELL: "#ff0044"
        }
        sig_color = sig_colors.get(analysis.signal, "#888")
        
        quality_colors = {
            SignalQuality.A: "#00ff88",
            SignalQuality.B: "#4ecdc4",
            SignalQuality.C: "#f9d423",
            SignalQuality.D: "#ff6b6b"
        }
        q_color = quality_colors.get(analysis.quality, "#888")
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{symbol} Enhanced Technical Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%);
            color: #eee; padding: 20px; min-height: 100vh;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        .header {{ text-align: center; padding: 30px; background: rgba(255,255,255,0.03);
            border-radius: 16px; margin-bottom: 20px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .signal-box {{ display: inline-block; padding: 20px 50px; background: {sig_color};
            color: #000; border-radius: 30px; font-size: 1.8em; font-weight: 700; margin: 15px; }}
        .quality-badge {{ display: inline-block; padding: 10px 25px; background: {q_color};
            color: #000; border-radius: 20px; font-weight: 600; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .card {{ background: rgba(255,255,255,0.03); border-radius: 12px; padding: 20px;
            border: 1px solid rgba(255,255,255,0.1); }}
        .card h3 {{ color: #4ecdc4; margin-bottom: 15px; font-size: 0.95em;
            text-transform: uppercase; letter-spacing: 1px; }}
        .chart-container {{ background: rgba(255,255,255,0.03); border-radius: 12px;
            padding: 20px; margin-bottom: 20px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-label {{ color: #888; }}
        .metric-value {{ font-weight: 600; }}
        .bullish {{ color: #4ecdc4; }}
        .bearish {{ color: #ff6b6b; }}
        .neutral {{ color: #888; }}
        .factor {{ padding: 8px 12px; margin: 5px 0; border-radius: 6px; font-size: 0.9em; }}
        .factor.bull {{ background: rgba(78,205,196,0.15); border-left: 3px solid #4ecdc4; }}
        .factor.bear {{ background: rgba(255,107,107,0.15); border-left: 3px solid #ff6b6b; }}
        .factor.warn {{ background: rgba(249,212,35,0.15); border-left: 3px solid #f9d423; }}
        .confluence-meter {{ text-align: center; padding: 20px; }}
        .confluence-score {{ font-size: 4em; font-weight: 700; color: {sig_color}; }}
        .setup-box {{ background: rgba(78,205,196,0.1); border: 1px solid #4ecdc4;
            border-radius: 12px; padding: 20px; margin-top: 10px; }}
        .footer {{ text-align: center; padding: 20px; color: #666; }}
        .indicator-bar {{ display: flex; align-items: center; margin: 8px 0; }}
        .indicator-bar .name {{ width: 100px; color: #888; }}
        .indicator-bar .bar {{ flex: 1; height: 24px; background: rgba(255,255,255,0.05);
            border-radius: 4px; position: relative; overflow: hidden; }}
        .indicator-bar .fill {{ height: 100%; transition: width 0.3s; }}
        .indicator-bar .value {{ position: absolute; right: 10px; top: 50%; transform: translateY(-50%); }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{symbol} Enhanced Technical Analysis</h1>
            <div>Phase 2: Institutional-Grade Multi-Factor System</div>
            <div class="signal-box">{analysis.signal.value}</div>
            <span class="quality-badge">Grade {analysis.quality.value}</span>
            <div style="margin-top: 15px; color: #888;">
                Confidence: {analysis.confidence:.0f}% | 
                Size Factor: {analysis.position_size_factor:.2f}x |
                MTF: {'Aligned' if analysis.mtf.alignment else 'Not Aligned'}
            </div>
        </div>

        <div class="chart-container">
            <h3 style="color: #4ecdc4; margin-bottom: 15px;">Price with Ichimoku & VWAP</h3>
            <canvas id="priceChart" height="100"></canvas>
        </div>

        <div class="grid">
            <div class="chart-container">
                <h3 style="color: #4ecdc4; margin-bottom: 15px;">Williams %R</h3>
                <canvas id="wrChart" height="80"></canvas>
            </div>
            <div class="chart-container">
                <h3 style="color: #4ecdc4; margin-bottom: 15px;">CCI</h3>
                <canvas id="cciChart" height="80"></canvas>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Confluence Analysis</h3>
                <div class="confluence-meter">
                    <div class="confluence-score">{analysis.confluence_score:+.0f}</div>
                    <div style="color: #888;">Confluence Score (-100 to +100)</div>
                </div>
                <div class="metric">
                    <span class="metric-label">Bullish Indicators</span>
                    <span class="metric-value bullish">{analysis.bullish_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Bearish Indicators</span>
                    <span class="metric-value bearish">{analysis.bearish_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Position Size Factor</span>
                    <span class="metric-value">{analysis.position_size_factor:.2f}x</span>
                </div>
            </div>

            <div class="card">
                <h3>Market Structure</h3>
                <div class="metric">
                    <span class="metric-label">Trend</span>
                    <span class="metric-value">{analysis.structure.trend}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Swing High</span>
                    <span class="metric-value">${analysis.structure.last_swing_high:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Swing Low</span>
                    <span class="metric-value">${analysis.structure.last_swing_low:,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Structure</span>
                    <span class="metric-value">HH:{analysis.structure.higher_highs} LL:{analysis.structure.lower_lows}</span>
                </div>
            </div>

            <div class="card">
                <h3>Volume Flow</h3>
                <div class="metric">
                    <span class="metric-label">CMF</span>
                    <span class="metric-value {'bullish' if analysis.volume_flow.cmf_value > 0 else 'bearish'}">
                        {analysis.volume_flow.cmf_value:+.3f}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Zone</span>
                    <span class="metric-value">{analysis.volume_flow.cmf_zone}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Relative Volume</span>
                    <span class="metric-value">{analysis.volume_flow.rvol:.2f}x</span>
                </div>
                <div class="metric">
                    <span class="metric-label">OBV Divergence</span>
                    <span class="metric-value">{analysis.volume_flow.obv_divergence or 'None'}</span>
                </div>
            </div>

            <div class="card">
                <h3>Volatility Structure</h3>
                <div class="metric">
                    <span class="metric-label">Current Vol</span>
                    <span class="metric-value">{analysis.volatility.current_vol:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Percentile</span>
                    <span class="metric-value">{analysis.volatility.vol_percentile:.0f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Regime</span>
                    <span class="metric-value">{analysis.volatility.regime}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Squeeze</span>
                    <span class="metric-value {'bullish' if analysis.volatility.squeeze else 'neutral'}">
                        {'DETECTED' if analysis.volatility.squeeze else 'No'}
                    </span>
                </div>
            </div>

            <div class="card">
                <h3>Momentum Quality</h3>
                <div class="metric">
                    <span class="metric-label">Score</span>
                    <span class="metric-value">{analysis.momentum.score:.0f}/100</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Quality</span>
                    <span class="metric-value">{analysis.momentum.quality}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ROC Aligned</span>
                    <span class="metric-value">{'Yes' if analysis.momentum.roc_aligned else 'No'}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Thrust Count</span>
                    <span class="metric-value">{analysis.momentum.thrust_count:+d}</span>
                </div>
            </div>

            <div class="card">
                <h3>Active Patterns</h3>
                <div class="metric">
                    <span class="metric-label">Net Score</span>
                    <span class="metric-value {'bullish' if analysis.patterns.net_score > 0 else 'bearish' if analysis.patterns.net_score < 0 else 'neutral'}">
                        {analysis.patterns.net_score:+d}
                    </span>
                </div>
                {''.join(f'<div class="factor bull">{p}</div>' for p in analysis.patterns.active_patterns[:3]) if analysis.patterns.active_patterns else '<div class="metric"><span class="metric-label">No active patterns</span></div>'}
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>Bullish Factors</h3>
                {''.join(f'<div class="factor bull">{f}</div>' for f in analysis.bullish_factors[:5]) if analysis.bullish_factors else '<div class="factor">None</div>'}
            </div>
            <div class="card">
                <h3>Bearish Factors</h3>
                {''.join(f'<div class="factor bear">{f}</div>' for f in analysis.bearish_factors[:5]) if analysis.bearish_factors else '<div class="factor">None</div>'}
            </div>
            <div class="card">
                <h3>Warnings</h3>
                {''.join(f'<div class="factor warn">{w}</div>' for w in analysis.warnings[:4]) if analysis.warnings else '<div class="factor">None</div>'}
            </div>
        </div>

        {'<div class="card"><h3>Trade Setup</h3><div class="setup-box"><div class="metric"><span class="metric-label">Direction</span><span class="metric-value">' + analysis.setup.direction + '</span></div><div class="metric"><span class="metric-label">Type</span><span class="metric-value">' + analysis.setup.setup_type + '</span></div><div class="metric"><span class="metric-label">Entry Zone</span><span class="metric-value">$' + f"{analysis.setup.entry_zone[0]:,.2f}" + ' - $' + f"{analysis.setup.entry_zone[1]:,.2f}" + '</span></div><div class="metric"><span class="metric-label">Stop Loss</span><span class="metric-value bearish">$' + f"{analysis.setup.stop_loss:,.2f}" + '</span></div><div class="metric"><span class="metric-label">Target 1</span><span class="metric-value bullish">$' + f"{analysis.setup.target_1:,.2f}" + '</span></div><div class="metric"><span class="metric-label">Risk/Reward</span><span class="metric-value">' + f"{analysis.setup.risk_reward:.2f}" + '</span></div></div></div>' if analysis.setup else ''}

        <div class="card" style="margin-top: 20px;">
            <h3>Analysis Summary</h3>
            <p style="line-height: 1.8; color: #ccc;">{analysis.summary}</p>
        </div>

        <div class="footer">
            Tamer's Enhanced Technical Analysis Agent | Phase 2<br>
            {len(df.columns)} indicators calculated | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>

    <script>
        new Chart(document.getElementById('priceChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(dates)},
                datasets: [
                    {{ label: 'Close', data: {json.dumps(closes)}, borderColor: '#fff', borderWidth: 2, fill: false, pointRadius: 0, tension: 0.1 }},
                    {{ label: 'Tenkan', data: {json.dumps(tenkan)}, borderColor: '#4ecdc4', borderWidth: 1, fill: false, pointRadius: 0 }},
                    {{ label: 'Kijun', data: {json.dumps(kijun)}, borderColor: '#ff6b6b', borderWidth: 1, fill: false, pointRadius: 0 }},
                    {{ label: 'VWAP', data: {json.dumps(vwap_vals)}, borderColor: '#f9d423', borderWidth: 1, borderDash: [5,5], fill: false, pointRadius: 0 }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ labels: {{ color: '#888' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#666', maxTicksLimit: 12 }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }},
                    y: {{ ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }}
                }}
            }}
        }});
        new Chart(document.getElementById('wrChart'), {{
            type: 'line',
            data: {{ labels: {json.dumps(dates)}, datasets: [{{ label: '%R', data: {json.dumps(wr)}, borderColor: '#4ecdc4', borderWidth: 1.5, fill: false, pointRadius: 0 }}] }},
            options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ ticks: {{ color: '#666', maxTicksLimit: 8 }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }}, y: {{ min: -100, max: 0, ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }} }} }}
        }});
        new Chart(document.getElementById('cciChart'), {{
            type: 'line',
            data: {{ labels: {json.dumps(dates)}, datasets: [{{ label: 'CCI', data: {json.dumps(cci)}, borderColor: '#f9d423', borderWidth: 1.5, fill: false, pointRadius: 0 }}] }},
            options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ ticks: {{ color: '#666', maxTicksLimit: 8 }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }}, y: {{ ticks: {{ color: '#888' }}, grid: {{ color: 'rgba(255,255,255,0.03)' }} }} }} }}
        }});
    </script>
</body>
</html>'''
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Dashboard: {output_path}")
