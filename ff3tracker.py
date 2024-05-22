# This only tracks the Fama French Factors using the following ETFs

import pandas as pd
import yfinance as yf

# CONSTANTS: TICKERS
MKT_REPLACEMENT = 'VTI'
RF_REPLACEMENT = '^IRX'
SMALL_REPLACEMENT = 'VB'
BIG_REPLACEMENT = 'VV'
VALUE_REPLACEMENT = 'VTV'
GROWTH_REPLACEMENT = 'VUG'

# CONSTANTS: DATE
HIST_START = '2005-01-01'

def generateFF3Tracker() -> pd.DataFrame:
  MKT_REPLACEMENT_HIST = yf.download(MKT_REPLACEMENT, start = HIST_START, end = None, auto_adjust=True)
  RF_REPLACEMENT_HIST = yf.download(RF_REPLACEMENT, start = HIST_START, end = None, auto_adjust=True)
  SMALL_REPLACEMENT_HIST = yf.download(SMALL_REPLACEMENT, start = HIST_START, end = None, auto_adjust=True)
  BIG_REPLACEMENT_HIST = yf.download(BIG_REPLACEMENT, start = HIST_START, end = None, auto_adjust=True)
  VALUE_REPLACEMENT_HIST = yf.download(VALUE_REPLACEMENT, start = HIST_START, end = None, auto_adjust=True)
  GROWTH_REPLACEMENT_HIST = yf.download(GROWTH_REPLACEMENT, start = HIST_START, end = None, auto_adjust=True)
  mkt = MKT_REPLACEMENT_HIST[['Close']].pct_change()[1:] * 100
  rf = ((RF_REPLACEMENT_HIST[['Close']] / 100 + 1) ** (1/252) - 1) * 100
  small = SMALL_REPLACEMENT_HIST[['Close']].pct_change()[1:] * 100
  big = BIG_REPLACEMENT_HIST[['Close']].pct_change()[1:] * 100
  value = VALUE_REPLACEMENT_HIST[['Close']].pct_change()[1:] * 100
  growth = GROWTH_REPLACEMENT_HIST[['Close']].pct_change()[1:] * 100

  replacements = pd.concat([mkt, rf, small, big, value, growth], axis=1, join='inner')
  replacements.columns=['MKT', 'RF', 'SMALL', 'BIG', 'VALUE', 'GROWTH']
  replacements['Mkt-RF'] = replacements['MKT'] - replacements['RF']
  replacements['SMB'] = replacements['SMALL'] - replacements['BIG']
  replacements['HML'] = replacements['VALUE'] - replacements['GROWTH']
  return replacements.reset_index()[['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']]