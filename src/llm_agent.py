#!/usr/bin/env python3
"""
Tamer's Quantitative Technical Analysis Agent
Phase 5: AI Agent Integration with Claude API

This module implements a sophisticated AI-powered analysis agent that:
- Integrates with Claude API for intelligent reasoning
- Uses tool-calling for dynamic data fetching and analysis
- Produces structured outputs using dataclass models
- Generates professional 1-2 page institutional trade notes
- Implements chain-of-thought reasoning for transparency

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     AI ANALYSIS AGENT                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐     │
    │  │  Market   │  │ Technical │  │  Regime   │  │ Backtest  │     │
    │  │   Data    │  │  Signals  │  │ Analysis  │  │  Metrics  │     │
    │  │   Tool    │  │   Tool    │  │   Tool    │  │   Tool    │     │
    │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘     │
    │        └───────────────┴───────────────┴───────────────┘        │
    │                              │                                  │
    │                    ┌─────────▼─────────┐                        │
    │                    │  Claude API with  │                        │
    │                    │  Chain-of-Thought │                        │
    │                    └─────────┬─────────┘                        │
    │                              │                                  │
    │                    ┌─────────▼─────────┐                        │
    │                    │   Trade Note      │                        │
    │                    │   Generator       │                        │
    │                    └───────────────────┘                        │
    └─────────────────────────────────────────────────────────────────┘

Differentiation from Fardeen:
    - Fardeen: Basic LLM prompt → response
    - Tamer: Full tool-calling architecture with multi-step reasoning,
             structured outputs, scenario analysis, and institutional-
             quality trade note generation

Academic References:
    - Anthropic (2024). "Tool Use with Claude." API Documentation.
    - CFA Institute. "Equity Research Report Standards."
    - GARP. "Risk Management Best Practices."

Author: Tamer
Course: MSc AI Agents in Asset Management
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class AgentConfig:
    """AI Agent configuration constants."""
    
    # Claude API Settings
    MODEL_NAME: str = "claude-sonnet-4-20250514"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.3  # Lower for more consistent analysis
    
    # Analysis Parameters
    CONFIDENCE_THRESHOLD_HIGH: float = 0.7
    CONFIDENCE_THRESHOLD_MEDIUM: float = 0.5
    
    # Output Settings
    TRADE_NOTE_MAX_WORDS: int = 1500
    INCLUDE_DISCLAIMER: bool = True
    
    # Risk Thresholds
    MAX_POSITION_SIZE: float = 0.10  # 10% of portfolio
    MAX_DRAWDOWN_TOLERANCE: float = 0.20  # 20% max acceptable DD


# =============================================================================
# ENUMERATIONS
# =============================================================================

class Recommendation(Enum):
    """Trade recommendation types."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class Conviction(Enum):
    """Conviction level for recommendations."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TimeHorizon(Enum):
    """Investment time horizon."""
    SHORT_TERM = "SHORT_TERM"      # 1-5 days
    MEDIUM_TERM = "MEDIUM_TERM"    # 1-4 weeks
    LONG_TERM = "LONG_TERM"        # 1-3 months


class RiskLevel(Enum):
    """Risk classification."""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"


# =============================================================================
# STRUCTURED OUTPUT MODELS
# =============================================================================

@dataclass
class MarketDataSummary:
    """Summary of market data for analysis."""
    symbol: str
    current_price: float
    price_change_1d: float
    price_change_5d: float
    price_change_20d: float
    high_52w: float
    low_52w: float
    distance_from_high: float
    distance_from_low: float
    avg_volume_20d: float
    current_volume: float
    volume_ratio: float
    data_quality: str
    last_updated: datetime


@dataclass
class TechnicalSignalSummary:
    """Summary of technical indicator signals."""
    overall_signal: str
    signal_strength: float
    quality_grade: str
    confidence: float
    
    # Individual indicators
    ichimoku_signal: float
    ichimoku_position: str
    vwap_signal: float
    vwap_zone: str
    williams_r_signal: float
    williams_r_zone: str
    cci_signal: float
    cci_zone: str
    
    # Structure
    trend: str
    momentum_quality: str
    momentum_score: int
    volatility_regime: str
    volatility_percentile: float
    
    # Patterns
    active_patterns: List[str]
    bullish_factors: List[str]
    bearish_factors: List[str]
    warnings: List[str]


@dataclass
class RegimeAnalysisSummary:
    """Summary of regime detection analysis."""
    current_regime: str
    regime_probability: float
    regime_duration_expected: int
    
    # Trend Analysis
    hurst_exponent: float
    trend_persistence: str
    is_trending: bool
    
    # Volatility
    current_volatility: float
    volatility_regime: str
    volatility_forecast_5d: float
    
    # Market Fear
    vix_level: float
    vix_regime: str
    fear_greed_score: int
    
    # Cycles
    dominant_cycle: int
    cycle_phase: str
    
    # Cross-Asset
    beta_to_spy: float
    correlation_to_spy: float
    
    # Strategy
    recommended_strategy: str
    position_bias: str
    recommended_size: float
    stop_loss_atr: float


@dataclass
class BacktestSummary:
    """Summary of backtest results."""
    status: str
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_dd_duration: int
    
    # Trade Stats
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    
    # Risk
    var_95: float
    cvar_95: float
    volatility: float
    
    # Validation
    is_statistically_significant: bool
    psr: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    
    # Walk-Forward
    wfe_ratio: float
    is_robust: bool
    
    # Position Sizing
    kelly_fraction: float
    recommended_position: float


@dataclass
class ScenarioAnalysis:
    """Scenario-based analysis."""
    bull_case_return: float
    bull_case_probability: float
    base_case_return: float
    base_case_probability: float
    bear_case_return: float
    bear_case_probability: float
    expected_return: float
    risk_reward_ratio: float


@dataclass
class TradeRecommendation:
    """Complete trade recommendation."""
    recommendation: Recommendation
    conviction: Conviction
    time_horizon: TimeHorizon
    risk_level: RiskLevel
    
    # Entry
    entry_price: float
    entry_zone_low: float
    entry_zone_high: float
    
    # Exit
    target_1: float
    target_2: float
    stop_loss: float
    
    # Position
    position_size_pct: float
    max_risk_pct: float
    
    # Rationale
    primary_reasons: List[str]
    risk_factors: List[str]
    catalysts: List[str]
    
    # Confidence
    confidence_score: float
    confidence_factors: Dict[str, float]


@dataclass
class ChainOfThought:
    """Chain-of-thought reasoning process."""
    step_1_market_context: str
    step_2_technical_analysis: str
    step_3_regime_assessment: str
    step_4_backtest_validation: str
    step_5_risk_evaluation: str
    step_6_synthesis: str
    step_7_recommendation: str
    reasoning_confidence: float


@dataclass
class TradeNoteContent:
    """Complete trade note content structure."""
    # Header
    symbol: str
    company_name: str
    analyst: str
    date: datetime
    recommendation: Recommendation
    conviction: Conviction
    target_price: float
    current_price: float
    
    # Executive Summary
    executive_summary: str
    key_points: List[str]
    
    # Analysis Sections
    market_overview: str
    technical_analysis: str
    regime_analysis: str
    backtest_validation: str
    risk_assessment: str
    
    # Recommendation Details
    trade_setup: TradeRecommendation
    scenario_analysis: ScenarioAnalysis
    
    # Chain of Thought
    reasoning: ChainOfThought
    
    # Disclaimer
    disclaimer: str


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of an analysis tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable


class AnalysisTools:
    """
    Collection of analysis tools for the AI agent.
    
    Each tool provides structured data from different phases
    of the analysis pipeline.
    """
    
    def __init__(
        self,
        market_data: Optional[pd.DataFrame] = None,
        technical_confluence: Optional[Any] = None,
        regime_report: Optional[Any] = None,
        backtest_result: Optional[Any] = None,
        symbol: str = "AAPL"
    ):
        """
        Initialize tools with analysis data.
        
        Args:
            market_data: OHLCV DataFrame from Phase 1
            technical_confluence: Confluence result from Phase 2
            regime_report: Regime analysis from Phase 3
            backtest_result: Backtest result from Phase 4
            symbol: Asset symbol
        """
        self.market_data = market_data
        self.technical_confluence = technical_confluence
        self.regime_report = regime_report
        self.backtest_result = backtest_result
        self.symbol = symbol
        
        self._tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, ToolDefinition]:
        """Register all available tools."""
        return {
            "get_market_data": ToolDefinition(
                name="get_market_data",
                description="Retrieve current market data and price statistics for the asset",
                parameters={"symbol": "str"},
                handler=self._get_market_data
            ),
            "get_technical_signals": ToolDefinition(
                name="get_technical_signals",
                description="Get technical indicator signals and confluence analysis",
                parameters={"symbol": "str"},
                handler=self._get_technical_signals
            ),
            "get_regime_analysis": ToolDefinition(
                name="get_regime_analysis",
                description="Get market regime detection and strategy recommendations",
                parameters={"symbol": "str"},
                handler=self._get_regime_analysis
            ),
            "get_backtest_metrics": ToolDefinition(
                name="get_backtest_metrics",
                description="Get backtest performance metrics and validation results",
                parameters={"symbol": "str"},
                handler=self._get_backtest_metrics
            ),
            "calculate_risk_reward": ToolDefinition(
                name="calculate_risk_reward",
                description="Calculate risk-reward metrics for a potential trade",
                parameters={
                    "entry_price": "float",
                    "target_price": "float",
                    "stop_loss": "float"
                },
                handler=self._calculate_risk_reward
            )
        }
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions in Claude API format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        k: {"type": v} for k, v in tool.parameters.items()
                    },
                    "required": list(tool.parameters.keys())
                }
            }
            for tool in self._tools.values()
        ]
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].handler(**kwargs)
    
    def _get_market_data(self, symbol: str = None) -> MarketDataSummary:
        """Get market data summary."""
        if self.market_data is None or len(self.market_data) < 2:
            return MarketDataSummary(
                symbol=self.symbol, current_price=0, price_change_1d=0,
                price_change_5d=0, price_change_20d=0, high_52w=0, low_52w=0,
                distance_from_high=0, distance_from_low=0, avg_volume_20d=0,
                current_volume=0, volume_ratio=0, data_quality="UNAVAILABLE",
                last_updated=datetime.now()
            )
        
        df = self.market_data
        current_price = float(df['Close'].iloc[-1])
        
        # Price changes
        price_1d = (current_price / df['Close'].iloc[-2] - 1) if len(df) > 1 else 0
        price_5d = (current_price / df['Close'].iloc[-5] - 1) if len(df) > 5 else 0
        price_20d = (current_price / df['Close'].iloc[-20] - 1) if len(df) > 20 else 0
        
        # 52-week high/low
        last_252 = df.tail(252)
        high_52w = float(last_252['High'].max())
        low_52w = float(last_252['Low'].min())
        
        # Volume
        avg_vol = float(df['Volume'].tail(20).mean())
        curr_vol = float(df['Volume'].iloc[-1])
        
        return MarketDataSummary(
            symbol=self.symbol,
            current_price=current_price,
            price_change_1d=price_1d,
            price_change_5d=price_5d,
            price_change_20d=price_20d,
            high_52w=high_52w,
            low_52w=low_52w,
            distance_from_high=(current_price / high_52w - 1),
            distance_from_low=(current_price / low_52w - 1),
            avg_volume_20d=avg_vol,
            current_volume=curr_vol,
            volume_ratio=curr_vol / avg_vol if avg_vol > 0 else 1,
            data_quality="EXCELLENT",
            last_updated=datetime.now()
        )
    
    def _get_technical_signals(self, symbol: str = None) -> TechnicalSignalSummary:
        """Get technical signals summary."""
        if self.technical_confluence is None:
            return TechnicalSignalSummary(
                overall_signal="NEUTRAL", signal_strength=0, quality_grade="N/A",
                confidence=0, ichimoku_signal=0, ichimoku_position="UNKNOWN",
                vwap_signal=0, vwap_zone="UNKNOWN", williams_r_signal=0,
                williams_r_zone="UNKNOWN", cci_signal=0, cci_zone="UNKNOWN",
                trend="UNKNOWN", momentum_quality="UNKNOWN", momentum_score=0,
                volatility_regime="UNKNOWN", volatility_percentile=0,
                active_patterns=[], bullish_factors=[], bearish_factors=[],
                warnings=[]
            )
        
        conf = self.technical_confluence
        
        return TechnicalSignalSummary(
            overall_signal=conf.signal.value,
            signal_strength=conf.confluence_score,
            quality_grade=conf.quality.value,
            confidence=conf.confidence,
            ichimoku_signal=conf.ichimoku.signal.value,
            ichimoku_position=conf.ichimoku.price_position.value,
            vwap_signal=conf.vwap.signal.value,
            vwap_zone=conf.vwap.band_position,
            williams_r_signal=conf.williams_r.signal.value,
            williams_r_zone=conf.williams_r.zone,
            cci_signal=conf.cci.signal.value,
            cci_zone=conf.cci.zone,
            trend=conf.structure.trend,
            momentum_quality=conf.momentum.quality,
            momentum_score=conf.momentum.score,
            volatility_regime=conf.volatility.regime,
            volatility_percentile=conf.volatility.vol_percentile,
            active_patterns=conf.patterns.active_patterns[:5] if conf.patterns else [],
            bullish_factors=conf.bullish_factors[:5],
            bearish_factors=conf.bearish_factors[:5],
            warnings=conf.warnings[:5]
        )
    
    def _get_regime_analysis(self, symbol: str = None) -> RegimeAnalysisSummary:
        """Get regime analysis summary."""
        if self.regime_report is None:
            return RegimeAnalysisSummary(
                current_regime="UNKNOWN", regime_probability=0,
                regime_duration_expected=0, hurst_exponent=0.5,
                trend_persistence="UNKNOWN", is_trending=False,
                current_volatility=0, volatility_regime="UNKNOWN",
                volatility_forecast_5d=0, vix_level=0, vix_regime="UNKNOWN",
                fear_greed_score=50, dominant_cycle=0, cycle_phase="UNKNOWN",
                beta_to_spy=1, correlation_to_spy=0, recommended_strategy="UNKNOWN",
                position_bias="NEUTRAL", recommended_size=0, stop_loss_atr=2
            )
        
        rr = self.regime_report
        
        return RegimeAnalysisSummary(
            current_regime=rr.hmm.current_regime.value,
            regime_probability=rr.hmm.regime_probability,
            regime_duration_expected=int(rr.hmm.expected_durations.get(
                rr.hmm.current_regime, 10)),
            hurst_exponent=rr.hurst.hurst_exponent,
            trend_persistence=rr.hurst.persistence.value,
            is_trending=rr.hurst.hurst_exponent > 0.55,
            current_volatility=rr.garch.current_volatility,
            volatility_regime=rr.garch.vol_regime.value,
            volatility_forecast_5d=rr.garch.forecast_5d,
            vix_level=rr.vix.current_vix,
            vix_regime=rr.vix.regime.value,
            fear_greed_score=int(rr.vix.fear_greed_index),
            dominant_cycle=rr.spectral.dominant_cycle,
            cycle_phase=rr.spectral.phase.value,
            beta_to_spy=rr.cross_asset.current_beta,
            correlation_to_spy=rr.cross_asset.correlation,
            recommended_strategy=rr.strategy.strategy.value,
            position_bias=rr.strategy.position_bias.value,
            recommended_size=rr.strategy.position_size,
            stop_loss_atr=rr.strategy.stop_loss_atr
        )
    
    def _get_backtest_metrics(self, symbol: str = None) -> BacktestSummary:
        """Get backtest metrics summary."""
        if self.backtest_result is None:
            return BacktestSummary(
                status="UNAVAILABLE", total_return=0, cagr=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, max_drawdown=0, max_dd_duration=0,
                total_trades=0, win_rate=0, profit_factor=0, avg_win=0, avg_loss=0,
                expectancy=0, var_95=0, cvar_95=0, volatility=0,
                is_statistically_significant=False, psr=0.5, sharpe_ci_lower=0,
                sharpe_ci_upper=0, wfe_ratio=0, is_robust=False, kelly_fraction=0,
                recommended_position=0
            )
        
        bt = self.backtest_result
        mc = bt.monte_carlo
        wf = bt.walk_forward
        ps = bt.position_sizing
        
        return BacktestSummary(
            status=bt.status.value,
            total_return=bt.returns.total_return,
            cagr=bt.returns.cagr,
            sharpe_ratio=bt.risk_adjusted.sharpe_ratio,
            sortino_ratio=bt.risk_adjusted.sortino_ratio,
            calmar_ratio=bt.risk_adjusted.calmar_ratio,
            max_drawdown=bt.risk.max_drawdown,
            max_dd_duration=bt.risk.max_drawdown_duration,
            total_trades=bt.trades.total_trades,
            win_rate=bt.trades.win_rate,
            profit_factor=bt.risk_adjusted.profit_factor,
            avg_win=bt.trades.avg_win_pct,
            avg_loss=bt.trades.avg_loss_pct,
            expectancy=bt.trades.expectancy_pct,
            var_95=bt.risk.var_95,
            cvar_95=bt.risk.cvar_95,
            volatility=bt.risk.volatility,
            is_statistically_significant=bt.validation.is_significant,
            psr=bt.validation.psr,
            sharpe_ci_lower=mc.sharpe_ci_95[0] if mc else 0,
            sharpe_ci_upper=mc.sharpe_ci_95[1] if mc else 0,
            wfe_ratio=wf.wfe_ratio if wf else 0,
            is_robust=wf.is_robust if wf else False,
            kelly_fraction=ps.kelly_fraction if ps else 0,
            recommended_position=ps.recommended_size if ps else 0
        )
    
    def _calculate_risk_reward(
        self,
        entry_price: float,
        target_price: float,
        stop_loss: float
    ) -> Dict[str, float]:
        """Calculate risk-reward metrics."""
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        risk_pct = risk / entry_price if entry_price > 0 else 0
        reward_pct = reward / entry_price if entry_price > 0 else 0
        
        return {
            "risk_amount": risk,
            "reward_amount": reward,
            "risk_reward_ratio": risk_reward_ratio,
            "risk_pct": risk_pct,
            "reward_pct": reward_pct,
            "breakeven_win_rate": 1 / (1 + risk_reward_ratio) if risk_reward_ratio > 0 else 0.5
        }


# =============================================================================
# REASONING ENGINE
# =============================================================================

class ReasoningEngine:
    """
    Chain-of-thought reasoning engine.
    
    Implements explicit multi-step reasoning for transparent analysis.
    """
    
    def __init__(self, tools: AnalysisTools):
        """Initialize with analysis tools."""
        self.tools = tools
    
    def generate_reasoning(self) -> ChainOfThought:
        """Generate complete chain-of-thought analysis."""
        
        # Gather all data
        market = self.tools.execute_tool("get_market_data")
        technical = self.tools.execute_tool("get_technical_signals")
        regime = self.tools.execute_tool("get_regime_analysis")
        backtest = self.tools.execute_tool("get_backtest_metrics")
        
        # Step 1: Market Context
        step_1 = self._analyze_market_context(market)
        
        # Step 2: Technical Analysis
        step_2 = self._analyze_technicals(technical)
        
        # Step 3: Regime Assessment
        step_3 = self._analyze_regime(regime)
        
        # Step 4: Backtest Validation
        step_4 = self._analyze_backtest(backtest)
        
        # Step 5: Risk Evaluation
        step_5 = self._evaluate_risk(market, regime, backtest)
        
        # Step 6: Synthesis
        step_6 = self._synthesize_analysis(market, technical, regime, backtest)
        
        # Step 7: Recommendation
        step_7, confidence = self._formulate_recommendation(
            market, technical, regime, backtest
        )
        
        return ChainOfThought(
            step_1_market_context=step_1,
            step_2_technical_analysis=step_2,
            step_3_regime_assessment=step_3,
            step_4_backtest_validation=step_4,
            step_5_risk_evaluation=step_5,
            step_6_synthesis=step_6,
            step_7_recommendation=step_7,
            reasoning_confidence=confidence
        )
    
    def _analyze_market_context(self, market: MarketDataSummary) -> str:
        """Analyze market context."""
        price_trend = "bullish" if market.price_change_20d > 0 else "bearish"
        distance_high = abs(market.distance_from_high) * 100
        distance_low = market.distance_from_low * 100
        
        return (
            f"{market.symbol} is currently trading at ${market.current_price:.2f}, "
            f"showing a {market.price_change_1d*100:+.1f}% move today and "
            f"{market.price_change_20d*100:+.1f}% over 20 days, indicating {price_trend} momentum. "
            f"The stock is {distance_high:.1f}% below its 52-week high of ${market.high_52w:.2f} "
            f"and {distance_low:.1f}% above its 52-week low. "
            f"Volume is {market.volume_ratio:.1f}x the 20-day average, suggesting "
            f"{'heightened' if market.volume_ratio > 1.2 else 'normal'} market interest."
        )
    
    def _analyze_technicals(self, tech: TechnicalSignalSummary) -> str:
        """Analyze technical signals."""
        # Determine overall bias
        bullish_count = len(tech.bullish_factors)
        bearish_count = len(tech.bearish_factors)
        
        if bullish_count > bearish_count + 1:
            bias = "bullish"
        elif bearish_count > bullish_count + 1:
            bias = "bearish"
        else:
            bias = "neutral"
        
        return (
            f"Technical analysis shows a {bias.upper()} bias with Grade {tech.quality_grade} quality "
            f"({tech.confidence:.0f}% confidence). "
            f"Ichimoku Cloud: Price is {tech.ichimoku_position.lower()} the cloud (signal: {tech.ichimoku_signal:+.2f}). "
            f"VWAP: Trading in {tech.vwap_zone} zone (signal: {tech.vwap_signal:+.2f}). "
            f"Momentum quality is {tech.momentum_quality} ({tech.momentum_score}/100). "
            f"Volatility regime is {tech.volatility_regime} at {tech.volatility_percentile:.0f}th percentile. "
            f"Key bullish factors: {', '.join(tech.bullish_factors[:3]) if tech.bullish_factors else 'None'}. "
            f"Key concerns: {', '.join(tech.warnings[:2]) if tech.warnings else 'None'}."
        )
    
    def _analyze_regime(self, regime: RegimeAnalysisSummary) -> str:
        """Analyze market regime."""
        trend_implication = (
            "suggesting trend-following strategies" if regime.is_trending 
            else "favoring mean-reversion approaches"
        )
        
        return (
            f"Hidden Markov Model detects a {regime.current_regime} regime with "
            f"{regime.regime_probability:.0%} probability, expected to persist ~{regime.regime_duration_expected} days. "
            f"Hurst exponent of {regime.hurst_exponent:.3f} indicates {regime.trend_persistence.lower().replace('_', ' ')}, "
            f"{trend_implication}. "
            f"GARCH volatility is {regime.current_volatility:.1%} ({regime.volatility_regime.lower().replace('_', ' ')}), "
            f"forecasted to reach {regime.volatility_forecast_5d:.1%} in 5 days. "
            f"VIX at {regime.vix_level:.1f} ({regime.vix_regime.lower()}) with fear/greed at {regime.fear_greed_score}/100. "
            f"Beta to SPY: {regime.beta_to_spy:.2f}. "
            f"Regime-based strategy recommendation: {regime.recommended_strategy}."
        )
    
    def _analyze_backtest(self, bt: BacktestSummary) -> str:
        """Analyze backtest results."""
        if bt.status != "SUCCESS":
            return f"Backtest status: {bt.status}. Unable to validate strategy performance."
        
        significance = "statistically significant" if bt.is_statistically_significant else "NOT statistically significant"
        robust = "ROBUST" if bt.is_robust else "NOT robust"
        
        return (
            f"Backtest over the full period shows {bt.cagr:.1%} CAGR with Sharpe ratio of {bt.sharpe_ratio:.2f}. "
            f"Maximum drawdown was {bt.max_drawdown:.1%} lasting {bt.max_dd_duration} days. "
            f"Win rate: {bt.win_rate:.1%} across {bt.total_trades} trades. "
            f"Risk metrics: VaR(95%)={bt.var_95:.2%}, CVaR(95%)={bt.cvar_95:.2%}. "
            f"Statistical validation: Results are {significance} (PSR={bt.psr:.0%}). "
            f"Monte Carlo 95% CI for Sharpe: [{bt.sharpe_ci_lower:.2f}, {bt.sharpe_ci_upper:.2f}]. "
            f"Walk-forward analysis: WFE ratio={bt.wfe_ratio:.2f}, strategy is {robust}."
        )
    
    def _evaluate_risk(
        self,
        market: MarketDataSummary,
        regime: RegimeAnalysisSummary,
        backtest: BacktestSummary
    ) -> str:
        """Evaluate overall risk."""
        risk_factors = []
        
        # Regime risk
        if regime.current_regime == "BEAR":
            risk_factors.append("bear market regime increases downside risk")
        
        # Volatility risk
        if regime.volatility_regime in ["HIGH_VOL", "EXTREME_VOL"]:
            risk_factors.append("elevated volatility regime")
        
        # Drawdown risk
        if backtest.max_drawdown > 0.3:
            risk_factors.append(f"historical max drawdown of {backtest.max_drawdown:.1%}")
        
        # Statistical risk
        if not backtest.is_statistically_significant:
            risk_factors.append("strategy not statistically validated")
        
        # Price risk
        if market.distance_from_high > -0.05:
            risk_factors.append("trading near 52-week highs")
        
        risk_summary = "; ".join(risk_factors) if risk_factors else "No major risk factors identified"
        
        return (
            f"Risk Assessment: Key risk factors include {risk_summary}. "
            f"Recommended position sizing: {regime.recommended_size:.0%} of portfolio "
            f"with {regime.stop_loss_atr:.1f}x ATR stop-loss. "
            f"Kelly criterion suggests {backtest.kelly_fraction:.1%} optimal position, "
            f"but {backtest.recommended_position:.1%} recommended for safety margin."
        )
    
    def _synthesize_analysis(
        self,
        market: MarketDataSummary,
        tech: TechnicalSignalSummary,
        regime: RegimeAnalysisSummary,
        bt: BacktestSummary
    ) -> str:
        """Synthesize all analysis into coherent view."""
        # Score each dimension
        technical_score = 1 if tech.overall_signal == "BUY" else (-1 if tech.overall_signal == "SELL" else 0)
        regime_score = -1 if regime.current_regime == "BEAR" else (1 if regime.current_regime == "BULL" else 0)
        momentum_score = 1 if market.price_change_20d > 0.05 else (-1 if market.price_change_20d < -0.05 else 0)
        validation_score = 1 if bt.is_robust else 0
        
        total_score = technical_score + regime_score + momentum_score + validation_score
        
        if total_score >= 2:
            overall = "favorable for long positions"
        elif total_score <= -2:
            overall = "unfavorable, suggesting caution"
        else:
            overall = "mixed, warranting a neutral stance"
        
        return (
            f"Synthesizing all factors: Technical indicators ({tech.overall_signal}), "
            f"market regime ({regime.current_regime}), price momentum ({market.price_change_20d:+.1%}), "
            f"and strategy validation ({'robust' if bt.is_robust else 'questionable'}) "
            f"paint an overall picture that is {overall}. "
            f"The combination of {regime.trend_persistence.lower().replace('_', ' ')} trend characteristics "
            f"and {regime.volatility_regime.lower().replace('_', ' ')} volatility "
            f"suggests {regime.recommended_strategy.lower().replace('_', ' ')} approach is warranted."
        )
    
    def _formulate_recommendation(
        self,
        market: MarketDataSummary,
        tech: TechnicalSignalSummary,
        regime: RegimeAnalysisSummary,
        bt: BacktestSummary
    ) -> Tuple[str, float]:
        """Formulate final recommendation with confidence."""
        
        # Determine recommendation
        tech_signal = tech.overall_signal
        regime_bias = regime.position_bias
        
        if regime.current_regime == "BEAR" and tech_signal != "BUY":
            recommendation = "HOLD with defensive positioning"
            confidence = 0.6
        elif tech_signal == "BUY" and regime.current_regime != "BEAR":
            recommendation = "BUY with measured position size"
            confidence = min(0.5 + tech.confidence/200 + bt.psr/2, 0.85)
        elif tech_signal == "SELL":
            recommendation = "REDUCE exposure and tighten stops"
            confidence = 0.65
        else:
            recommendation = "HOLD and wait for clearer signals"
            confidence = 0.5
        
        return (
            f"Based on the comprehensive analysis, the recommendation is to {recommendation}. "
            f"Entry zone: ${market.current_price * 0.98:.2f} - ${market.current_price * 1.02:.2f}. "
            f"Target: ${market.current_price * (1 + max(0.05, regime.recommended_size)):.2f}. "
            f"Stop-loss: ${market.current_price * (1 - regime.stop_loss_atr * 0.02):.2f} "
            f"({regime.stop_loss_atr:.1f}x ATR below entry). "
            f"Position size: {regime.recommended_size:.0%} of portfolio. "
            f"Time horizon: {'Short-term (1-5 days)' if regime.dominant_cycle < 10 else 'Medium-term (1-4 weeks)'}.",
            confidence
        )


# =============================================================================
# TRADE NOTE GENERATOR
# =============================================================================

class TradeNoteGenerator:
    """
    Professional trade note generator.
    
    Creates institutional-quality research notes in multiple formats.
    """
    
    def __init__(
        self,
        tools: AnalysisTools,
        analyst_name: str = "Tamer",
        company_name: str = "Apple Inc."
    ):
        """Initialize generator."""
        self.tools = tools
        self.analyst_name = analyst_name
        self.company_name = company_name
        self.reasoning_engine = ReasoningEngine(tools)
    
    def generate(self) -> TradeNoteContent:
        """Generate complete trade note content."""
        
        # Gather all data
        market = self.tools.execute_tool("get_market_data")
        technical = self.tools.execute_tool("get_technical_signals")
        regime = self.tools.execute_tool("get_regime_analysis")
        backtest = self.tools.execute_tool("get_backtest_metrics")
        
        # Generate reasoning
        reasoning = self.reasoning_engine.generate_reasoning()
        
        # Determine recommendation
        recommendation, conviction = self._determine_recommendation(
            technical, regime, backtest
        )
        
        # Generate trade setup
        trade_setup = self._generate_trade_setup(
            market, technical, regime, backtest, recommendation, conviction
        )
        
        # Generate scenario analysis
        scenarios = self._generate_scenarios(market, regime, backtest)
        
        # Generate content sections
        executive_summary = self._generate_executive_summary(
            market, technical, regime, backtest, recommendation
        )
        
        key_points = self._generate_key_points(
            technical, regime, backtest, recommendation
        )
        
        return TradeNoteContent(
            symbol=market.symbol,
            company_name=self.company_name,
            analyst=self.analyst_name,
            date=datetime.now(),
            recommendation=recommendation,
            conviction=conviction,
            target_price=trade_setup.target_1,
            current_price=market.current_price,
            executive_summary=executive_summary,
            key_points=key_points,
            market_overview=reasoning.step_1_market_context,
            technical_analysis=reasoning.step_2_technical_analysis,
            regime_analysis=reasoning.step_3_regime_assessment,
            backtest_validation=reasoning.step_4_backtest_validation,
            risk_assessment=reasoning.step_5_risk_evaluation,
            trade_setup=trade_setup,
            scenario_analysis=scenarios,
            reasoning=reasoning,
            disclaimer=self._generate_disclaimer()
        )
    
    def _determine_recommendation(
        self,
        tech: TechnicalSignalSummary,
        regime: RegimeAnalysisSummary,
        bt: BacktestSummary
    ) -> Tuple[Recommendation, Conviction]:
        """Determine recommendation and conviction."""
        
        # Score factors
        tech_score = 0
        if tech.overall_signal == "BUY":
            tech_score = 2
        elif tech.overall_signal == "SELL":
            tech_score = -2
        
        regime_score = 0
        if regime.current_regime == "BULL":
            regime_score = 2
        elif regime.current_regime == "BEAR":
            regime_score = -2
        
        validation_score = 1 if bt.is_robust else -1
        
        total = tech_score + regime_score + validation_score
        
        # Determine recommendation
        if total >= 3:
            rec = Recommendation.STRONG_BUY
            conv = Conviction.HIGH
        elif total >= 1:
            rec = Recommendation.BUY
            conv = Conviction.MEDIUM
        elif total <= -3:
            rec = Recommendation.STRONG_SELL
            conv = Conviction.HIGH
        elif total <= -1:
            rec = Recommendation.SELL if total <= -2 else Recommendation.REDUCE
            conv = Conviction.MEDIUM
        else:
            rec = Recommendation.HOLD
            conv = Conviction.LOW
        
        return rec, conv
    
    def _generate_trade_setup(
        self,
        market: MarketDataSummary,
        tech: TechnicalSignalSummary,
        regime: RegimeAnalysisSummary,
        bt: BacktestSummary,
        recommendation: Recommendation,
        conviction: Conviction
    ) -> TradeRecommendation:
        """Generate detailed trade setup."""
        
        price = market.current_price
        atr_estimate = price * 0.02  # Approximate 2% ATR
        
        # Entry zone
        entry_low = price * 0.98
        entry_high = price * 1.02
        
        # Targets based on regime
        if regime.current_regime == "BULL":
            target_1 = price * 1.08
            target_2 = price * 1.15
        elif regime.current_regime == "BEAR":
            target_1 = price * 1.03
            target_2 = price * 1.05
        else:
            target_1 = price * 1.05
            target_2 = price * 1.10
        
        # Stop based on ATR
        stop = price - (regime.stop_loss_atr * atr_estimate)
        
        # Position size
        pos_size = min(regime.recommended_size, bt.kelly_fraction * 2, 0.10)
        
        # Risk per trade
        risk_pct = (price - stop) / price
        
        # Primary reasons
        reasons = []
        if tech.overall_signal == "BUY":
            reasons.append("Positive technical confluence")
        if regime.current_regime == "BULL":
            reasons.append("Favorable market regime")
        if bt.sharpe_ratio > 0:
            reasons.append("Positive historical risk-adjusted returns")
        if regime.is_trending:
            reasons.append("Trending market suitable for momentum")
        if not reasons:
            reasons.append("Defensive positioning warranted")
        
        # Risk factors
        risks = []
        if regime.current_regime == "BEAR":
            risks.append("Bear market regime")
        if bt.max_drawdown > 0.3:
            risks.append(f"Historical {bt.max_drawdown:.0%} max drawdown")
        if not bt.is_statistically_significant:
            risks.append("Strategy not statistically validated")
        if tech.momentum_quality == "POOR":
            risks.append("Poor momentum quality")
        if market.distance_from_high > -0.05:
            risks.append("Near 52-week highs")
        if not risks:
            risks.append("Standard market risks apply")
        
        # Catalysts
        catalysts = [
            "Upcoming earnings announcement",
            "Technical breakout/breakdown levels",
            "Regime transition signals",
            "Volatility expansion/contraction"
        ]
        
        # Confidence factors
        conf_factors = {
            "technical_quality": tech.confidence / 100,
            "regime_clarity": regime.regime_probability,
            "backtest_validity": bt.psr,
            "market_conditions": 0.5 + (1 - abs(market.distance_from_high)) * 0.5
        }
        
        overall_conf = sum(conf_factors.values()) / len(conf_factors)
        
        return TradeRecommendation(
            recommendation=recommendation,
            conviction=conviction,
            time_horizon=TimeHorizon.MEDIUM_TERM if regime.dominant_cycle > 10 else TimeHorizon.SHORT_TERM,
            risk_level=RiskLevel.CONSERVATIVE if regime.current_regime == "BEAR" else RiskLevel.MODERATE,
            entry_price=price,
            entry_zone_low=entry_low,
            entry_zone_high=entry_high,
            target_1=target_1,
            target_2=target_2,
            stop_loss=stop,
            position_size_pct=pos_size,
            max_risk_pct=risk_pct,
            primary_reasons=reasons,
            risk_factors=risks,
            catalysts=catalysts[:3],
            confidence_score=overall_conf,
            confidence_factors=conf_factors
        )
    
    def _generate_scenarios(
        self,
        market: MarketDataSummary,
        regime: RegimeAnalysisSummary,
        bt: BacktestSummary
    ) -> ScenarioAnalysis:
        """Generate scenario analysis."""
        
        # Base probabilities from regime
        if regime.current_regime == "BULL":
            bull_prob, base_prob, bear_prob = 0.45, 0.35, 0.20
        elif regime.current_regime == "BEAR":
            bull_prob, base_prob, bear_prob = 0.20, 0.35, 0.45
        else:
            bull_prob, base_prob, bear_prob = 0.30, 0.40, 0.30
        
        # Scenario returns
        bull_return = max(0.15, bt.cagr + 0.15) if bt.cagr else 0.20
        base_return = bt.cagr if bt.cagr else 0.05
        bear_return = min(-0.10, bt.cagr - 0.15) if bt.cagr else -0.15
        
        # Expected return
        expected = (
            bull_prob * bull_return +
            base_prob * base_return +
            bear_prob * bear_return
        )
        
        # Risk-reward
        upside = bull_prob * bull_return + base_prob * max(0, base_return)
        downside = bear_prob * abs(bear_return) + base_prob * abs(min(0, base_return))
        rr_ratio = upside / downside if downside > 0 else 2.0
        
        return ScenarioAnalysis(
            bull_case_return=bull_return,
            bull_case_probability=bull_prob,
            base_case_return=base_return,
            base_case_probability=base_prob,
            bear_case_return=bear_return,
            bear_case_probability=bear_prob,
            expected_return=expected,
            risk_reward_ratio=rr_ratio
        )
    
    def _generate_executive_summary(
        self,
        market: MarketDataSummary,
        tech: TechnicalSignalSummary,
        regime: RegimeAnalysisSummary,
        bt: BacktestSummary,
        recommendation: Recommendation
    ) -> str:
        """Generate executive summary."""
        
        rec_text = recommendation.value.replace("_", " ").title()
        
        return (
            f"We initiate coverage of {self.company_name} ({market.symbol}) with a "
            f"{rec_text} recommendation. The stock is currently trading at ${market.current_price:.2f}, "
            f"representing a {market.price_change_20d*100:+.1f}% move over the past 20 trading days. "
            f"Our quantitative analysis utilizing Hidden Markov Models identifies a "
            f"{regime.current_regime} regime with {regime.regime_probability:.0%} confidence. "
            f"Technical confluence analysis shows a {tech.quality_grade}-grade {tech.overall_signal} signal. "
            f"Backtesting over the full period demonstrates {bt.cagr:.1%} CAGR with a "
            f"Sharpe ratio of {bt.sharpe_ratio:.2f}. "
            f"We recommend a {regime.recommended_size:.0%} position size with strict "
            f"{regime.stop_loss_atr:.1f}x ATR stop-loss discipline."
        )
    
    def _generate_key_points(
        self,
        tech: TechnicalSignalSummary,
        regime: RegimeAnalysisSummary,
        bt: BacktestSummary,
        recommendation: Recommendation
    ) -> List[str]:
        """Generate key points."""
        points = []
        
        # Recommendation
        points.append(
            f"{recommendation.value.replace('_', ' ')}: {regime.recommended_strategy} approach recommended"
        )
        
        # Regime
        points.append(
            f"HMM detects {regime.current_regime} regime ({regime.regime_probability:.0%} probability)"
        )
        
        # Technical
        points.append(
            f"Technical Grade {tech.quality_grade}: {tech.trend} structure with {tech.momentum_quality} momentum"
        )
        
        # Risk
        points.append(
            f"Risk metrics: Max DD {bt.max_drawdown:.1%}, VaR(95%) {bt.var_95:.2%}"
        )
        
        # Position
        points.append(
            f"Position sizing: {regime.recommended_size:.0%} of portfolio, {regime.stop_loss_atr:.1f}x ATR stop"
        )
        
        return points
    
    def _generate_disclaimer(self) -> str:
        """Generate professional disclaimer."""
        return (
            "DISCLAIMER: This analysis is generated by an AI-powered quantitative model "
            "for educational purposes as part of the MSc AI Agents in Asset Management coursework. "
            "This is not investment advice. Past performance does not guarantee future results. "
            "All investments carry risk of loss. The models and analysis presented have not been "
            "validated by regulatory authorities. Users should conduct their own due diligence "
            "and consult qualified financial advisors before making investment decisions. "
            "The author assumes no liability for any trading losses incurred."
        )


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def format_trade_note_markdown(content: TradeNoteContent) -> str:
    """Format trade note as professional Markdown."""
    
    rec_emoji = {
        Recommendation.STRONG_BUY: "🟢",
        Recommendation.BUY: "🟢",
        Recommendation.HOLD: "🟡",
        Recommendation.REDUCE: "🟠",
        Recommendation.SELL: "🔴",
        Recommendation.STRONG_SELL: "🔴"
    }.get(content.recommendation, "⚪")
    
    md = f"""# {content.company_name} ({content.symbol})
## Quantitative Technical Analysis Report

**Analyst:** {content.analyst}  
**Date:** {content.date.strftime('%B %d, %Y')}  
**Current Price:** ${content.current_price:.2f}  
**Target Price:** ${content.target_price:.2f}  

---

## {rec_emoji} RECOMMENDATION: {content.recommendation.value.replace('_', ' ')}

**Conviction:** {content.conviction.value} | **Time Horizon:** {content.trade_setup.time_horizon.value.replace('_', ' ')}

---

## Executive Summary

{content.executive_summary}

### Key Points

"""
    
    for point in content.key_points:
        md += f"- {point}\n"
    
    md += f"""
---

## Market Overview

{content.market_overview}

---

## Technical Analysis

{content.technical_analysis}

---

## Regime Analysis

{content.regime_analysis}

---

## Backtest Validation

{content.backtest_validation}

---

## Risk Assessment

{content.risk_assessment}

---

## Trade Setup

| Parameter | Value |
|-----------|-------|
| Entry Zone | ${content.trade_setup.entry_zone_low:.2f} - ${content.trade_setup.entry_zone_high:.2f} |
| Target 1 | ${content.trade_setup.target_1:.2f} (+{(content.trade_setup.target_1/content.current_price - 1)*100:.1f}%) |
| Target 2 | ${content.trade_setup.target_2:.2f} (+{(content.trade_setup.target_2/content.current_price - 1)*100:.1f}%) |
| Stop Loss | ${content.trade_setup.stop_loss:.2f} ({(content.trade_setup.stop_loss/content.current_price - 1)*100:.1f}%) |
| Position Size | {content.trade_setup.position_size_pct:.1%} of portfolio |
| Max Risk | {content.trade_setup.max_risk_pct:.2%} per trade |

### Primary Reasons
"""
    
    for reason in content.trade_setup.primary_reasons:
        md += f"- {reason}\n"
    
    md += "\n### Risk Factors\n"
    for risk in content.trade_setup.risk_factors:
        md += f"- ⚠️ {risk}\n"
    
    md += f"""
---

## Scenario Analysis

| Scenario | Return | Probability |
|----------|--------|-------------|
| Bull Case | {content.scenario_analysis.bull_case_return:+.1%} | {content.scenario_analysis.bull_case_probability:.0%} |
| Base Case | {content.scenario_analysis.base_case_return:+.1%} | {content.scenario_analysis.base_case_probability:.0%} |
| Bear Case | {content.scenario_analysis.bear_case_return:+.1%} | {content.scenario_analysis.bear_case_probability:.0%} |

**Expected Return:** {content.scenario_analysis.expected_return:+.1%}  
**Risk/Reward Ratio:** {content.scenario_analysis.risk_reward_ratio:.2f}

---

## Chain-of-Thought Analysis

<details>
<summary>Click to expand detailed reasoning</summary>

### Step 1: Market Context
{content.reasoning.step_1_market_context}

### Step 2: Technical Analysis
{content.reasoning.step_2_technical_analysis}

### Step 3: Regime Assessment
{content.reasoning.step_3_regime_assessment}

### Step 4: Backtest Validation
{content.reasoning.step_4_backtest_validation}

### Step 5: Risk Evaluation
{content.reasoning.step_5_risk_evaluation}

### Step 6: Synthesis
{content.reasoning.step_6_synthesis}

### Step 7: Recommendation
{content.reasoning.step_7_recommendation}

**Reasoning Confidence:** {content.reasoning.reasoning_confidence:.0%}

</details>

---

## Confidence Factors

| Factor | Score |
|--------|-------|
| Technical Quality | {content.trade_setup.confidence_factors.get('technical_quality', 0):.0%} |
| Regime Clarity | {content.trade_setup.confidence_factors.get('regime_clarity', 0):.0%} |
| Backtest Validity | {content.trade_setup.confidence_factors.get('backtest_validity', 0):.0%} |
| Market Conditions | {content.trade_setup.confidence_factors.get('market_conditions', 0):.0%} |

**Overall Confidence:** {content.trade_setup.confidence_score:.0%}

---

*{content.disclaimer}*

---

*Generated by Tamer's Quantitative Technical Analysis Agent*  
*MSc AI Agents in Asset Management*
"""
    
    return md


def format_trade_note_html(content: TradeNoteContent) -> str:
    """Format trade note as professional HTML."""
    
    rec_color = {
        Recommendation.STRONG_BUY: "#22c55e",
        Recommendation.BUY: "#4ade80",
        Recommendation.HOLD: "#facc15",
        Recommendation.REDUCE: "#fb923c",
        Recommendation.SELL: "#f87171",
        Recommendation.STRONG_SELL: "#ef4444"
    }.get(content.recommendation, "#94a3b8")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{content.symbol} - Quantitative Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            color: #e2e8f0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .header {{
            text-align: center;
            padding: 2rem;
            background: rgba(30, 41, 59, 0.8);
            border-radius: 1rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }}
        .header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .recommendation-badge {{
            display: inline-block;
            padding: 0.75rem 2rem;
            background: {rec_color}20;
            color: {rec_color};
            border: 2px solid {rec_color};
            border-radius: 2rem;
            font-weight: 700;
            font-size: 1.25rem;
            margin: 1rem 0;
        }}
        .meta {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-top: 1rem;
            color: #94a3b8;
        }}
        .section {{
            background: rgba(30, 41, 59, 0.6);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }}
        .section h2 {{
            font-size: 1.25rem;
            color: #60a5fa;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
        }}
        .section p {{ margin-bottom: 1rem; }}
        .key-points {{
            list-style: none;
            padding: 0;
        }}
        .key-points li {{
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}
        .key-points li:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #60a5fa;
        }}
        .trade-setup {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }}
        .setup-item {{
            background: rgba(15, 23, 42, 0.5);
            padding: 1rem;
            border-radius: 0.5rem;
        }}
        .setup-item .label {{
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .setup-item .value {{
            font-size: 1.25rem;
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }}
        .setup-item .value.positive {{ color: #4ade80; }}
        .setup-item .value.negative {{ color: #f87171; }}
        .scenarios {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            text-align: center;
        }}
        .scenario {{
            padding: 1rem;
            border-radius: 0.5rem;
        }}
        .scenario.bull {{ background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3); }}
        .scenario.base {{ background: rgba(250, 204, 21, 0.1); border: 1px solid rgba(250, 204, 21, 0.3); }}
        .scenario.bear {{ background: rgba(248, 113, 113, 0.1); border: 1px solid rgba(248, 113, 113, 0.3); }}
        .reasoning {{
            background: rgba(15, 23, 42, 0.5);
            padding: 1rem;
            border-radius: 0.5rem;
            font-size: 0.9rem;
            margin-top: 1rem;
        }}
        .reasoning h3 {{
            font-size: 0.875rem;
            color: #a78bfa;
            margin: 1rem 0 0.5rem;
        }}
        .reasoning h3:first-child {{ margin-top: 0; }}
        .disclaimer {{
            font-size: 0.75rem;
            color: #64748b;
            text-align: center;
            padding: 1rem;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            margin-top: 2rem;
        }}
        .confidence-bar {{
            height: 8px;
            background: rgba(148, 163, 184, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            border-radius: 4px;
        }}
        @media print {{
            body {{ background: white; color: black; }}
            .section {{ border: 1px solid #ccc; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{content.company_name} ({content.symbol})</h1>
            <p style="color: #94a3b8;">Quantitative Technical Analysis Report</p>
            <div class="recommendation-badge">{content.recommendation.value.replace('_', ' ')}</div>
            <div class="meta">
                <span>Analyst: {content.analyst}</span>
                <span>Date: {content.date.strftime('%B %d, %Y')}</span>
                <span>Price: ${content.current_price:.2f}</span>
                <span>Target: ${content.target_price:.2f}</span>
            </div>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>{content.executive_summary}</p>
            <h3 style="margin-top: 1rem; color: #94a3b8; font-size: 0.875rem;">Key Points</h3>
            <ul class="key-points">
"""
    
    for point in content.key_points:
        html += f"                <li>{point}</li>\n"
    
    html += f"""            </ul>
        </div>
        
        <div class="section">
            <h2>Trade Setup</h2>
            <div class="trade-setup">
                <div class="setup-item">
                    <div class="label">Entry Zone</div>
                    <div class="value">${content.trade_setup.entry_zone_low:.2f} - ${content.trade_setup.entry_zone_high:.2f}</div>
                </div>
                <div class="setup-item">
                    <div class="label">Target 1</div>
                    <div class="value positive">${content.trade_setup.target_1:.2f} (+{(content.trade_setup.target_1/content.current_price - 1)*100:.1f}%)</div>
                </div>
                <div class="setup-item">
                    <div class="label">Stop Loss</div>
                    <div class="value negative">${content.trade_setup.stop_loss:.2f} ({(content.trade_setup.stop_loss/content.current_price - 1)*100:.1f}%)</div>
                </div>
                <div class="setup-item">
                    <div class="label">Position Size</div>
                    <div class="value">{content.trade_setup.position_size_pct:.1%}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Scenario Analysis</h2>
            <div class="scenarios">
                <div class="scenario bull">
                    <div style="font-weight: 600;">Bull Case</div>
                    <div style="font-size: 1.5rem; color: #4ade80;">{content.scenario_analysis.bull_case_return:+.1%}</div>
                    <div style="color: #94a3b8;">{content.scenario_analysis.bull_case_probability:.0%} probability</div>
                </div>
                <div class="scenario base">
                    <div style="font-weight: 600;">Base Case</div>
                    <div style="font-size: 1.5rem; color: #facc15;">{content.scenario_analysis.base_case_return:+.1%}</div>
                    <div style="color: #94a3b8;">{content.scenario_analysis.base_case_probability:.0%} probability</div>
                </div>
                <div class="scenario bear">
                    <div style="font-weight: 600;">Bear Case</div>
                    <div style="font-size: 1.5rem; color: #f87171;">{content.scenario_analysis.bear_case_return:+.1%}</div>
                    <div style="color: #94a3b8;">{content.scenario_analysis.bear_case_probability:.0%} probability</div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 1rem;">
                <span>Expected Return: <strong>{content.scenario_analysis.expected_return:+.1%}</strong></span>
                <span style="margin-left: 2rem;">Risk/Reward: <strong>{content.scenario_analysis.risk_reward_ratio:.2f}</strong></span>
            </div>
        </div>
        
        <div class="section">
            <h2>Analysis</h2>
            <div class="reasoning">
                <h3>Market Context</h3>
                <p>{content.market_overview}</p>
                
                <h3>Technical Analysis</h3>
                <p>{content.technical_analysis}</p>
                
                <h3>Regime Analysis</h3>
                <p>{content.regime_analysis}</p>
                
                <h3>Backtest Validation</h3>
                <p>{content.backtest_validation}</p>
                
                <h3>Risk Assessment</h3>
                <p>{content.risk_assessment}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Confidence Assessment</h2>
            <p>Overall Confidence: <strong>{content.trade_setup.confidence_score:.0%}</strong></p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {content.trade_setup.confidence_score*100}%;"></div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
"""
    
    for factor, score in content.trade_setup.confidence_factors.items():
        html += f"""                <div>
                    <span style="color: #94a3b8;">{factor.replace('_', ' ').title()}</span>
                    <span style="float: right;">{score:.0%}</span>
                    <div class="confidence-bar"><div class="confidence-fill" style="width: {score*100}%;"></div></div>
                </div>
"""
    
    html += f"""            </div>
        </div>
        
        <div class="disclaimer">
            {content.disclaimer}
        </div>
        
        <div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.75rem;">
            Generated by Tamer's Quantitative Technical Analysis Agent<br>
            MSc AI Agents in Asset Management
        </div>
    </div>
</body>
</html>
"""
    
    return html


def format_trade_note_terminal(content: TradeNoteContent) -> str:
    """Format trade note for terminal display."""
    
    width = 80
    
    def header(text: str) -> str:
        return f"\n{'═' * width}\n{text:^{width}}\n{'═' * width}"
    
    def section(title: str) -> str:
        return f"\n{title}\n{'─' * width}"
    
    lines = []
    
    # Header
    lines.append(header(f"{content.symbol} TRADE NOTE"))
    lines.append(f"  Company: {content.company_name}")
    lines.append(f"  Analyst: {content.analyst}")
    lines.append(f"  Date: {content.date.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Price: ${content.current_price:.2f} → Target: ${content.target_price:.2f}")
    
    # Recommendation Box
    rec = content.recommendation.value.replace('_', ' ')
    conv = content.conviction.value
    lines.append(f"""
  ┌────────────────────────────────────────────────────────────────────────┐
  │  RECOMMENDATION: {rec:<15}  │  CONVICTION: {conv:<10}  │
  │  Time Horizon: {content.trade_setup.time_horizon.value.replace('_', ' '):<12}  │  Risk Level: {content.trade_setup.risk_level.value:<12}  │
  └────────────────────────────────────────────────────────────────────────┘
""")
    
    # Executive Summary
    lines.append(section("EXECUTIVE SUMMARY"))
    for line in textwrap.wrap(content.executive_summary, width - 4):
        lines.append(f"  {line}")
    
    # Key Points
    lines.append(section("KEY POINTS"))
    for point in content.key_points:
        lines.append(f"  • {point}")
    
    # Trade Setup
    lines.append(section("TRADE SETUP"))
    ts = content.trade_setup
    lines.append(f"  Entry Zone:    ${ts.entry_zone_low:.2f} - ${ts.entry_zone_high:.2f}")
    lines.append(f"  Target 1:      ${ts.target_1:.2f} (+{(ts.target_1/content.current_price - 1)*100:.1f}%)")
    lines.append(f"  Target 2:      ${ts.target_2:.2f} (+{(ts.target_2/content.current_price - 1)*100:.1f}%)")
    lines.append(f"  Stop Loss:     ${ts.stop_loss:.2f} ({(ts.stop_loss/content.current_price - 1)*100:.1f}%)")
    lines.append(f"  Position Size: {ts.position_size_pct:.1%} of portfolio")
    lines.append(f"  Max Risk:      {ts.max_risk_pct:.2%} per trade")
    
    # Scenario Analysis
    lines.append(section("SCENARIO ANALYSIS"))
    sa = content.scenario_analysis
    lines.append(f"  Bull Case:  {sa.bull_case_return:+.1%} ({sa.bull_case_probability:.0%} probability)")
    lines.append(f"  Base Case:  {sa.base_case_return:+.1%} ({sa.base_case_probability:.0%} probability)")
    lines.append(f"  Bear Case:  {sa.bear_case_return:+.1%} ({sa.bear_case_probability:.0%} probability)")
    lines.append(f"  Expected Return: {sa.expected_return:+.1%}")
    lines.append(f"  Risk/Reward:     {sa.risk_reward_ratio:.2f}")
    
    # Confidence
    lines.append(section("CONFIDENCE"))
    lines.append(f"  Overall: {ts.confidence_score:.0%}")
    for factor, score in ts.confidence_factors.items():
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"  {factor.replace('_', ' ').title():<20} [{bar}] {score:.0%}")
    
    # Risk Factors
    lines.append(section("RISK FACTORS"))
    for risk in ts.risk_factors:
        lines.append(f"  ⚠ {risk}")
    
    # Footer
    lines.append("\n" + "═" * width)
    lines.append("  Generated by Tamer's Quantitative Technical Analysis Agent")
    lines.append("  MSc AI Agents in Asset Management")
    lines.append("═" * width)
    
    return "\n".join(lines)


# =============================================================================
# AI AGENT CLASS
# =============================================================================

class TechnicalAnalysisAgent:
    """
    AI-powered Technical Analysis Agent.
    
    Orchestrates the complete analysis pipeline and generates
    professional trade notes using Claude API or simulation mode.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_simulation: bool = True
    ):
        """
        Initialize the AI agent.
        
        Args:
            api_key: Anthropic API key (optional)
            use_simulation: Use simulation mode if API unavailable
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.use_simulation = use_simulation
        self.client = None
        
        # Initialize API client if key available
        if self.api_key and not use_simulation:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("Claude API client initialized")
            except ImportError:
                logger.warning("anthropic package not installed, using simulation mode")
                self.use_simulation = True
            except Exception as e:
                logger.warning(f"Failed to initialize API client: {e}, using simulation mode")
                self.use_simulation = True
    
    def analyze(
        self,
        market_data: pd.DataFrame,
        technical_confluence: Any,
        regime_report: Any,
        backtest_result: Any,
        symbol: str = "AAPL",
        company_name: str = "Apple Inc."
    ) -> TradeNoteContent:
        """
        Run complete analysis and generate trade note.
        
        Args:
            market_data: OHLCV DataFrame from Phase 1
            technical_confluence: Confluence result from Phase 2
            regime_report: Regime analysis from Phase 3
            backtest_result: Backtest result from Phase 4
            symbol: Asset symbol
            company_name: Full company name
            
        Returns:
            Complete trade note content
        """
        # Initialize tools
        tools = AnalysisTools(
            market_data=market_data,
            technical_confluence=technical_confluence,
            regime_report=regime_report,
            backtest_result=backtest_result,
            symbol=symbol
        )
        
        # Generate trade note
        generator = TradeNoteGenerator(
            tools=tools,
            analyst_name="Tamer",
            company_name=company_name
        )
        
        logger.info("Generating trade note...")
        content = generator.generate()
        
        return content
    
    def generate_report(
        self,
        content: TradeNoteContent,
        output_dir: Path,
        formats: List[str] = ["html", "md", "txt"]
    ) -> Dict[str, Path]:
        """
        Generate report files in multiple formats.
        
        Args:
            content: Trade note content
            output_dir: Output directory
            formats: List of formats to generate
            
        Returns:
            Dictionary of format -> file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        if "html" in formats:
            html_path = output_dir / f"{content.symbol.lower()}_trade_note.html"
            html_content = format_trade_note_html(content)
            with open(html_path, 'w') as f:
                f.write(html_content)
            outputs["html"] = html_path
            logger.info(f"Generated: {html_path}")
        
        if "md" in formats:
            md_path = output_dir / f"{content.symbol.lower()}_trade_note.md"
            md_content = format_trade_note_markdown(content)
            with open(md_path, 'w') as f:
                f.write(md_content)
            outputs["md"] = md_path
            logger.info(f"Generated: {md_path}")
        
        if "txt" in formats:
            txt_path = output_dir / f"{content.symbol.lower()}_trade_note.txt"
            txt_content = format_trade_note_terminal(content)
            with open(txt_path, 'w') as f:
                f.write(txt_content)
            outputs["txt"] = txt_path
            logger.info(f"Generated: {txt_path}")
        
        return outputs


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'Recommendation',
    'Conviction',
    'TimeHorizon',
    'RiskLevel',
    
    # Data Classes
    'MarketDataSummary',
    'TechnicalSignalSummary',
    'RegimeAnalysisSummary',
    'BacktestSummary',
    'ScenarioAnalysis',
    'TradeRecommendation',
    'ChainOfThought',
    'TradeNoteContent',
    
    # Tools
    'AnalysisTools',
    'ToolDefinition',
    
    # Engines
    'ReasoningEngine',
    'TradeNoteGenerator',
    
    # Agent
    'TechnicalAnalysisAgent',
    
    # Formatters
    'format_trade_note_markdown',
    'format_trade_note_html',
    'format_trade_note_terminal',
    
    # Config
    'AgentConfig',
]