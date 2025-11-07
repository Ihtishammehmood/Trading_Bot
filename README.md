# Forex Backtesting with Metatrader5 and Backtrader

![YouTube Thumbnail](img/trading_thumnail.PNG)

# [Watch Full Video](https://youtu.be/KtFDI2TQdRE)
## Overview

A backtesting project combining MetaTrader 5 for data feeds and Backtrader for strategy implementation and testing.
## Setup
- Pip install `uv`
- clone Repository
- create virtual Environment `uv venv` and activate the environment `.venv\Scripts\activate` on windows
- Install Dependencies `uv sync`
## Core Components

### Data Ingestion (`data_ingest.py`)

- MetaTrader 5 integration for forex data
- Supports custom timeframes and date ranges
- Automatic data formatting for Backtrader compatibility

### Trading Strategies

1. **SMA Strategy**
   - Uses Simple Moving Average crossover
   - Default period: 20
   - Entry: Price crosses above SMA
   - Exit: Price crosses below SMA

2. **Bollinger Bands Strategy**
   - Parameters: 20-period, 2 standard deviations
   - Entry: Price crosses below lower band
   - Exit: Price crosses above upper band
   - Uses market open execution

### Utilities

- Custom trade logging system
- Specialized buy/sell indicators
- Performance tracking and reporting

## Project Dependencies

- Python 3.12+
- MetaTrader5
- Backtrader
- Pandas
- Additional packages in pyproject.toml

## Key Features

- Real forex data integration
- Commission handling (0.1%)
- Portfolio value tracking
- Custom trade visualization
- Detailed execution logging
