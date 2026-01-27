#!/usr/bin/env python3
"""
Tamer's Quantitative Technical Analysis Agent
Phase 1 + Phase 2 Demo

Phase 1: Institutional Data Pipeline
- Multi-asset acquisition (AAPL + SPY + VIX)
- Earnings calendar integration for Anchored VWAP
- Multi-timeframe aggregation (Daily/Weekly/Monthly)

Phase 2: Advanced Technical Indicators
- Ichimoku Cloud (5-component system)
- VWAP + Anchored VWAP (earnings-based)
- Williams %R with divergence detection
- CCI with extreme readings
- Confluence scoring

Author: Tamer
Course: MSc AI Agents in Asset Management
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


BANNER = r'''
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║      ████████╗ █████╗ ███╗   ███╗███████╗██████╗                              ║
║      ╚══██╔══╝██╔══██╗████╗ ████║██╔════╝██╔══██╗                             ║
║         ██║   ███████║██╔████╔██║█████╗  ██████╔╝                             ║
║         ██║   ██╔══██║██║╚██╔╝██║██╔══╝  ██╔══██╗                             ║
║         ██║   ██║  ██║██║ ╚═╝ ██║███████╗██║  ██║                             ║
║         ╚═╝   ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝                             ║
║                                                                               ║
║              QUANTITATIVE TECHNICAL ANALYSIS AGENT                            ║
║                                                                               ║
║              MSc AI Agents in Asset Management                                ║
║              Phase 1: Institutional Data Pipeline                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
'''

ARCHITECTURE = '''
┌───────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA PIPELINE ARCHITECTURE                       │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐│
│   │   ACQUIRE   │────▶│   ENRICH    │────▶│  AGGREGATE  │────▶│   EXPORT    ││
│   │ AAPL+SPY    │     │ Returns     │     │ Daily       │     │ Parquet     ││
│   │ +VIX        │     │ TrueRange   │     │ Weekly      │     │ JSON        ││
│   │ +Earnings   │     │ TypicalP    │     │ Monthly     │     │ HTML        ││
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘│
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                   PHASE 2: TECHNICAL INDICATOR ARCHITECTURE                   │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐│
│   │  ICHIMOKU   │     │    VWAP     │     │ WILLIAMS %R │     │     CCI     ││
│   │  CLOUD      │     │   SUITE     │     │  ADVANCED   │     │ PROFESSIONAL││
│   │             │     │             │     │             │     │             ││
│   │ • 5 Lines   │     │ • Standard  │     │ • Dual 7/21 │     │ • Dual 14/50││
│   │ • TK Cross  │     │ • Rolling   │     │ • Regular   │     │ • Zero Cross││
│   │ • Kumo Brk  │     │   5D/20D    │     │   Diverge   │     │ • Divergence││
│   │ • Kijun Bnc │     │ • Anchored  │     │ • Hidden    │     │ • Hooks     ││
│   │ • Cloud Twt │     │   Earnings  │     │   Diverge   │     │ • Zones     ││
│   │ • MTF Align │     │ • Inst Zone │     │ • FailSwing │     │ • Trend     ││
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘│
│          │                   │                   │                   │       │
│          └───────────────────┴───────────────────┴───────────────────┘       │
│                                      │                                       │
│                    ┌─────────────────▼─────────────────┐                     │
│                    │     MULTI-TIMEFRAME CONFLUENCE    │                     │
│                    │                                   │                     │
│                    │  • Daily + Weekly Alignment       │                     │
│                    │  • Weighted Signal Scoring        │                     │
│                    │  • Quality Grading (A/B/C/D)      │                     │
│                    │  • Trade Setup Identification     │                     │
│                    └───────────────────────────────────┘                     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
'''

ROADMAP = '''
┌───────────────────────────────────────────────────────────────────────────────┐
│                           IMPLEMENTATION ROADMAP                              │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ■■■■■■■■■■  Phase 1: Data Pipeline                            [COMPLETE]    │
│              • Multi-asset batch fetch (AAPL + SPY + VIX)                     │
│              • Earnings calendar for Anchored VWAP                            │
│              • Multi-timeframe aggregation (D/W/M)                            │
│              • Statistical profiling with strategy hints                      │
│                                                                               │
│  ■■■■■■■■■■  Phase 2: Advanced Technical Indicators            [COMPLETE]    │
│              • Ichimoku Cloud (5 components + TK cross + Kumo breakout)       │
│              • VWAP Suite (Standard + Rolling 5D/20D + Anchored earnings)     │
│              • Williams %R Advanced (Dual 7/21 + Regular/Hidden divergence)   │
│              • CCI Professional (Dual 14/50 + Hooks + Divergence)             │
│              • Multi-Timeframe Confluence (Daily/Weekly alignment)            │
│              • Signal Quality Grading (A/B/C/D) + Trade Setup ID              │
│                                                                               │
│  ░░░░░░░░░░  Phase 3: Regime Detection                         [PENDING]     │
│              • Hidden Markov Model (Bull/Bear/Sideways)                       │
│              • Hurst Exponent (trend persistence)                             │
│              • VIX-based volatility regime overlay                            │
│                                                                               │
│  ░░░░░░░░░░  Phase 4: Backtesting & Metrics                    [PENDING]     │
│              • VectorBT with transaction costs                                │
│              • Walk-forward optimization (5:1 IS/OOS)                         │
│              • Monte Carlo (10,000 block bootstrap simulations)               │
│              • Sortino, Calmar, Omega, PSR, VaR, CVaR                         │
│                                                                               │
│  ░░░░░░░░░░  Phase 5: AI Agent Integration                     [PENDING]     │
│              • Claude API with structured outputs                             │
│              • Tool-calling for dynamic analysis                              │
│              • 1-2 page professional trade note                               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
'''


def main() -> int:
    """Execute Phase 1 and Phase 2 demonstration."""
    print(BANNER)
    print(f"  Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Target: Apple Inc. (AAPL)")
    print("  Benchmark: S&P 500 ETF (SPY)")
    print("  Volatility: CBOE VIX Index (^VIX)")
    print("  Period: 10 years (2015-01-01 to 2025-12-31)")
    print()
    print(ARCHITECTURE)
    
    logging.basicConfig(
        level=logging.INFO,
        format="  %(asctime)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # ==========================================================================
    # PHASE 1: DATA PIPELINE
    # ==========================================================================
    
    print("═" * 79)
    print("  PHASE 1: DATA PIPELINE")
    print("═" * 79)
    print()
    
    try:
        from data_collector import DataPipeline, print_report, QualityGrade
        
        pipeline = DataPipeline("config/config.yaml")
        phase1_output = pipeline.run()
        
        print_report(phase1_output)
        
        if phase1_output.quality.grade not in [QualityGrade.EXCELLENT, QualityGrade.GOOD]:
            print(f"  Phase 1 quality issues: {phase1_output.quality.grade.value}")
            return 1
        
        # Get earnings dates for Phase 2
        earnings_dates = [e.date for e in phase1_output.earnings]
    
    except Exception as e:
        print(f"\n  PHASE 1 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ==========================================================================
    # PHASE 2: TECHNICAL INDICATORS
    # ==========================================================================
    
    print()
    print("═" * 79)
    print("  PHASE 2: ENHANCED TECHNICAL INDICATORS")
    print("═" * 79)
    print()
    
    try:
        from technical_indicators import (
            EnhancedTechnicalAnalyzer, 
            IndicatorDashboard, 
            print_enhanced_analysis
        )
        
        # Initialize enhanced analyzer with config
        config = {}
        try:
            import yaml
            with open("config/config.yaml") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            pass
        
        analyzer = EnhancedTechnicalAnalyzer(config.get('indicators', {}))
        
        # Run enhanced analysis on Phase 1 data (daily + weekly)
        df_indicators, weekly_indicators, confluence = analyzer.analyze(
            phase1_output.daily,
            phase1_output.weekly,
            earnings_dates=earnings_dates
        )
        
        # Save enriched data
        indicators_path = Path("data/aapl_indicators.parquet")
        df_indicators.to_parquet(indicators_path, compression='snappy')
        logging.getLogger(__name__).info(f"Saved: {indicators_path}")
        
        # Generate dashboard
        dashboard_path = Path("outputs/reports/aapl_indicators.html")
        IndicatorDashboard.generate(
            df_indicators, 
            confluence, 
            pipeline.symbol, 
            dashboard_path
        )
        
        # Print enhanced analysis report
        print_enhanced_analysis(confluence, pipeline.symbol)
        
    except Exception as e:
        print(f"\n  PHASE 2 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    
    print(ROADMAP)
    
    print("═" * 79)
    print("  EXECUTION COMPLETE")
    print("═" * 79)
    
    p = phase1_output.profile
    ich = confluence.ichimoku
    vw = confluence.vwap
    wr = confluence.williams_r
    cc = confluence.cci
    struct = confluence.structure
    vol = confluence.volatility
    mom = confluence.momentum
    vf = confluence.volume_flow
    
    print(f"""
  Phase 1 - Data Pipeline:
    • Data Quality: {phase1_output.quality.overall:.1f}/100 ({phase1_output.quality.grade.value})
    • Daily Records: {len(phase1_output.daily):,}
    • Weekly Records: {len(phase1_output.weekly):,}
    • Earnings Events: {len(phase1_output.earnings)}
    
  Phase 2 - Enhanced Technical Analysis:
    ┌────────────────────────────────────────────────────────────────────────┐
    │  SIGNAL: {confluence.signal.value:^14}  │  GRADE: {confluence.quality.value:^6}  │  CONF: {confluence.confidence:>3.0f}%  │
    │  Confluence Score: {confluence.confluence_score:>+5.0f}   │  Position Size: {confluence.position_size_factor:.2f}x       │
    └────────────────────────────────────────────────────────────────────────┘
    
    Core Indicators:
      Ichimoku:    {ich.signal.value:+.2f} │ {ich.price_position.value:12} │ Cloud: {ich.cloud_color.value}
      VWAP:        {vw.signal.value:+.2f} │ {vw.band_position:12} │ Dist: {vw.distance_pct:+.1f}%
      Williams %R: {wr.signal.value:+.2f} │ Fast: {wr.fast_value:>5.1f}     │ {wr.zone}
      CCI:         {cc.signal.value:+.2f} │ Short: {cc.short_value:>4.0f}    │ {cc.zone}
    
    Advanced Analysis:
      Structure:   {struct.trend:13} │ HH:{struct.higher_highs} LL:{struct.lower_lows}
      Volume Flow: {vf.cmf_zone:13} │ CMF: {vf.cmf_value:+.3f} │ RVOL: {vf.rvol:.2f}x
      Volatility:  {vol.regime:13} │ {vol.current_vol:.1f}% ann │ Pctl: {vol.vol_percentile:.0f}%
      Momentum:    {mom.quality:13} │ Score: {mom.score:.0f}/100
    
    MTF Alignment: {'Yes' if confluence.mtf.alignment else 'No'} ({confluence.mtf.daily_trend.value} / {confluence.mtf.weekly_trend.value})
    """)
    
    if confluence.patterns.active_patterns:
        print(f"    Active Patterns: {', '.join(confluence.patterns.active_patterns[:3])}")
    
    if confluence.setup:
        print(f"""
    Trade Setup:
      Direction:   {confluence.setup.direction}
      Type:        {confluence.setup.setup_type}
      Entry:       ${confluence.setup.entry_zone[0]:,.2f} - ${confluence.setup.entry_zone[1]:,.2f}
      Stop:        ${confluence.setup.stop_loss:,.2f}
      Target:      ${confluence.setup.target_1:,.2f}
      R/R:         {confluence.setup.risk_reward:.2f}
    """)
    
    if confluence.warnings:
        print("    Warnings:")
        for w in confluence.warnings[:3]:
            print(f"      ⚠ {w}")
    
    print(f"""
  Exports:
    • data/aapl_daily.parquet
    • data/aapl_weekly.parquet
    • data/aapl_indicators.parquet ({len(df_indicators.columns)} columns)
    • outputs/reports/aapl_report.html (Phase 1)
    • outputs/reports/aapl_indicators.html (Phase 2) ← OPEN THIS!

  Next: Phase 3 (Regime Detection with HMM and Hurst Exponent)
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
