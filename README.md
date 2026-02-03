# Technical Analyst Agent

MSc AI Agents in Asset Management
IFTE0001 Coursework - Track B


## Project Overview

A multi-phase technical analysis pipeline that generates institutional-grade trade recommendations. The system processes historical market data through quantitative analysis, backtesting validation, and LLM-powered narrative generation to produce actionable trade notes with comprehensive JSON output for orchestration.


## System Architecture

```
Phase 1: Data Pipeline
    - OHLCV data acquisition via yfinance
    - Data quality validation and scoring
    - Statistical tests (ADF, KPSS, Jarque-Bera)
    - Benchmark correlation analysis
    - Volatility estimators (Parkinson, Garman-Klass, Yang-Zhang)

Phase 2: Technical Indicators
    - Momentum: RSI, Stochastic, Williams %R
    - Trend: MACD, ADX, Supertrend, Ichimoku
    - Volatility: Bollinger Bands, Keltner Channels, ATR
    - Volume: OBV, MFI, CMF
    - Weighted signal aggregation with confidence scoring

Phase 3: Market Regime Detection
    - Hidden Markov Model (3-state: Bull, Bear, Sideways)
    - GARCH(1,1) volatility modeling
    - Hurst exponent analysis
    - Structural break detection (CUSUM)
    - Strategy recommendation engine

Phase 4: Backtesting Engine
    - Signal-based strategy execution
    - Walk-forward optimization (5 periods)
    - Monte Carlo simulation (500 paths)
    - Transaction cost modeling (10 bps)
    - Performance metrics calculation

Phase 4B: Risk Analytics
    - CAPM alpha/beta decomposition
    - Regime-conditional performance
    - Signal quality assessment
    - Historical stress testing
    - Drawdown analysis

Phase 5: LLM Trade Notes
    - Claude Sonnet 4.5 integration
    - Investment thesis generation
    - Multi-format output (JSON, PDF, MD, HTML, TXT)
    - 247-field structured data for orchestration
```


## Installation

1. Create and activate virtual environment:

```
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Configure API key:

```
export ANTHROPIC_API_KEY="your-api-key"
```


## Usage

Basic execution:

```
python run_demo.py --symbol AAPL
```

Command line options:

```
--symbol        Stock ticker symbol (default: AAPL)
--years         Years of historical data (default: 10)
--output-dir    Output directory path (default: ./output)
--skip-llm      Skip LLM trade note generation
--verbose       Enable verbose logging
```

Examples:

```
python run_demo.py --symbol MSFT --years 5
python run_demo.py --symbol GOOGL --skip-llm
python run_demo.py --symbol NVDA --output-dir ./reports --verbose
```


## Project Structure

```
src/
    data_collector.py         Phase 1: Data acquisition and validation
    technical_indicators.py   Phase 2: Technical indicator computation
    regime_detector.py        Phase 3: Market regime classification
    backtest_engine.py        Phase 4: Strategy backtesting
    risk_analytics.py         Phase 4B: Risk attribution analysis
    llm_agent.py              Phase 5: Trade note generation
    report_generator.py       Multi-format report output
    trade_note_reports.py     PDF report generation
    config.py                 Configuration parameters

run_demo.py                   Main execution script
requirements.txt              Python dependencies
README.md                     Documentation
```


## Output Files

```
output/
    {SYMBOL}_trade_note.json    Structured data (247 fields)
    {SYMBOL}_trade_note.pdf     Professional report
    {SYMBOL}_trade_note.md      Markdown summary
    {SYMBOL}_trade_note.html    Interactive web report
    {SYMBOL}_trade_note.txt     Plain text version
```


## JSON Output Structure

The JSON output contains 247 fields organized into the following categories:

Identification (11 fields)
- note_id, symbol, company_name, sector, industry
- generated_at, analysis_period, total_records

Recommendation (5 fields)
- recommendation (BUY/SELL/HOLD)
- conviction (HIGH/MEDIUM/LOW)
- confidence (0-100)
- time_horizon
- risk_rating

Trade Setup (13 fields)
- entry, stop_loss
- target_1, target_2, target_3, target_price
- risk_reward, position_size, max_size
- sizing_method, sizing_rationale

Technical Scores (9 fields)
- overall_score, technical_score
- momentum_score, trend_score, volatility_score, volume_score
- risk_score
- score_explanations, score_breakdown

Backtest Metrics (24 fields)
- backtest_cagr, backtest_sharpe, backtest_sortino, backtest_calmar
- backtest_max_dd, backtest_hit_rate, backtest_profit_factor
- backtest_total_trades, backtest_avg_win, backtest_avg_loss
- backtest_expectancy, backtest_total_return, backtest_volatility

Risk Metrics (7 fields)
- var_95, var_99
- cvar_95, cvar_99
- var_95_strategy, cvar_95_strategy

Phase 3: Regime Detection (37 fields)
- market_regime, regime_probability, regime_confidence
- volatility_regime, current_volatility, volatility_percentile
- garch_omega, garch_alpha, garch_beta, garch_persistence
- volatility_forecast_1d, volatility_forecast_5d
- hurst_p3, hurst_classification, hurst_r_squared
- structural_breaks, days_since_break
- strategy_recommendation, position_bias, strategy_confidence

Phase 4B: Risk Attribution (52 fields)
- alpha, beta, r_squared, alpha_t_stat, alpha_p_value
- tracking_error, information_ratio, skill_contribution
- performance_grade, risk_level, alpha_confidence, strategy_robustness
- perf_bull_return, perf_bull_sharpe, perf_bear_return, perf_bear_sharpe
- stress test results for COVID crash, 2022 bear, 2018 Q4

Narratives (8 fields)
- investment_thesis
- executive_summary
- technical_analysis
- backtest_analysis
- risk_analysis
- scenarios (bull/base/bear with probabilities)
- catalysts
- risks


## Coursework Requirements

Required Metrics:
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted)
- Calmar Ratio (drawdown-adjusted)
- Maximum Drawdown
- Hit Rate (win percentage)
- Profit Factor (gross profit / gross loss)
- VaR 95% and 99% (Value at Risk)
- CVaR 95% and 99% (Conditional VaR)

LLM Trade Notes:
- Investment thesis with quantitative rationale
- Executive summary with actionable recommendation
- Technical analysis narrative
- Backtest performance analysis
- Risk assessment with mitigation strategies
- Scenario analysis (bull/base/bear cases)


## Configuration

Key parameters in config.py:

```
Data Collection:
    DEFAULT_YEARS = 10
    DATA_SOURCE = "yfinance"

Technical Indicators:
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2.0

Backtesting:
    INITIAL_CAPITAL = 100000
    TRANSACTION_COST = 0.001
    SLIPPAGE = 0.0005

Risk Management:
    MAX_POSITION_SIZE = 0.10
    STOP_LOSS_ATR = 2.0
    TAKE_PROFIT_ATR = 4.0

LLM Configuration:
    CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
    MAX_TOKENS = 12000
    TEMPERATURE = 0.3
```


## Module Reference

data_collector.py:
```
from data_collector import DataCollector
collector = DataCollector()
result = collector.collect(symbol="AAPL", years=10)
```

technical_indicators.py:
```
from technical_indicators import TechnicalAnalyzer
analyzer = TechnicalAnalyzer()
result = analyzer.analyze(df, symbol="AAPL")
```

regime_detector.py:
```
from regime_detector import RegimeDetector
detector = RegimeDetector()
result = detector.analyze(df, symbol="AAPL")
```

backtest_engine.py:
```
from backtest_engine import BacktestEngine
engine = BacktestEngine()
result = engine.run(df, signals, symbol="AAPL")
```

risk_analytics.py:
```
from risk_analytics import RiskAnalyticsEngine
engine = RiskAnalyticsEngine()
result = engine.analyze(strategy_returns, benchmark_returns, signals, regimes)
```

llm_agent.py:
```
from llm_agent import generate_trade_note
note = generate_trade_note(p1, p2, p3, p4, p4b)
json_output = note.to_dict()
```


## Version History

9.2.2  Fixed Phase 3/4B dataclass attribute extraction
9.2.1  Fixed stress test None handling
9.2.0  Added 247-field comprehensive JSON output
9.1.2  Fixed PDF layout issues
9.1.1  Fixed VaR/CVaR extraction from Phase 4
9.1.0  Initial coursework submission


## Author

MSc AI Agents in Asset Management
IFTE0001 Coursework - Track B: Technical Analyst Agent
