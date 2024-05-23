from typing import Union

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from pypfopt import black_litterman

import MVO
import utils


class Analyst:
  """Analyst Class - Retrieve confidence levels about every analyst"""
  def __init__(self, ratings: pd.DataFrame, historical_data: pd.DataFrame) -> None:
    # Get ratings data
    ratings = ratings[['OFTIC', 'ANNDATS', 'AMASKCD', 'IRECCD']]
    # Disregard NEUTRAL ratings
    ratings = ratings[ratings['IRECCD'] != '3']
    ratings['IRECCD'] = ratings['IRECCD'] > "3"
    ratings.columns = ['OFTIC', 'DATE', 'AMASKCD', 'IRECCD']
  
    # Get full historical data (Large File)
    hist_data = historical_data[['DATE', 'TICKER', 'PRC']]
    hist_data = hist_data.dropna()
    hist_data = hist_data.drop_duplicates(subset=['DATE', 'TICKER'])
    # Process the data to prepare for merge
    hist_data = hist_data.pivot(columns='TICKER', index='DATE', values='PRC')
    hist_data = hist_data.resample('1d').mean().ffill()
    
    # Preprocess historical data to identify growth or decline (monthly)
    bool_growth = (hist_data - hist_data.shift(periods=-30)).applymap(lambda x: np.NaN if np.isnan(x) else x > 0)
    bool_growth = bool_growth.melt(var_name='OFTIC',value_name='IS_GROW', ignore_index=False)
    bool_growth.dropna(inplace=True)
    bool_growth.reset_index(names=['DATE'], inplace=True)
    
    # Merge the two datasets
    self.ratings = pd.merge(ratings, bool_growth, on=['OFTIC', 'DATE'], how='inner')
  
  def rank(self, end_date: Union[str, None] = None) -> pd.DataFrame:
    """Get confidence levels without look ahead bias - confidence prior to end_date"""
    data = self.ratings
    if end_date is not None:
      data = self.ratings[self.ratings['DATE'] <= end_date]
    grouped = data.groupby('AMASKCD')
    analysts = {'AMASKCD': [], 'CONFIDENCE': []}
    for name, group in grouped:
      analysts['AMASKCD'].append(name)
      conf = (group['IRECCD'] == group['IS_GROW']).sum() / len(group)
      if len(group) <= 1:
        conf = 0.5
      analysts['CONFIDENCE'].append(conf)
    return pd.DataFrame(analysts)
  
class PriceTarget:
  """Price Target Class - Retrieves look ahead bias adjusted price targets (priors) for selected stocks"""
  def __init__(self, price_targets: pd.DataFrame, rankings: pd.DataFrame, tickers: list) -> None:
    self.tickers = tickers
    
    pt_masked = price_targets[['OFTIC', 'ANNDATS', 'AMASKCD', 'CURR', 'ESTCUR', 'HORIZON', 'VALUE']]
    pt_masked = pt_masked[(pt_masked['CURR'] == 'USD') & (pt_masked['ESTCUR'] == 'USD')]
    pt_masked['HORIZON'] = pt_masked['HORIZON'].astype(int)
    pt_masked = pt_masked[(pt_masked['HORIZON'] >= 1) & (pt_masked['HORIZON'] <= 12)]
    pt_masked = pt_masked[['OFTIC', 'ANNDATS', 'AMASKCD', 'HORIZON', 'VALUE']]
    pt_masked.columns = ['OFTIC', 'DATE', 'AMASKCD', 'HORIZON', 'VALUE']

    pt_merged = pd.merge(pt_masked, rankings, on=['AMASKCD'], how='left')
    pt_merged['CONFIDENCE'] = pt_merged['CONFIDENCE'].fillna(0.5)
    self.price_targets = pt_merged[pt_merged['OFTIC'].isin(tickers)]
    
  def get_views(self, pf: MVO.Portfolio, end_date: str, confidence_multiplier = 0.1):
    """Note pf should be a portfolio that ends on end_date"""
    # We only want views made in the last month
    pt_filtered = self.price_targets
    pt_filtered = pt_filtered[(pt_filtered['DATE'] <= end_date) & (pt_filtered['DATE'] >= utils.dateadd(end_date, "m", -1))]
    
    last_day = pd.DataFrame(pf.data['Close'].iloc[-1]).reset_index()
    last_day.columns = ['OFTIC', 'CLOSE']
    last_day
    
    pt_pct = pd.merge(pt_filtered, last_day, on='OFTIC')
    pt_pct['RET'] = (pt_pct['VALUE'] - pt_pct['CLOSE']) / pt_pct['CLOSE'] / pt_pct['HORIZON']
    pt_pct['CONFIDENCE'] = pt_pct['CONFIDENCE'] * confidence_multiplier
    
    # Black Litterman Functions - Uses the pypfopt package for ease
    bl_P = pd.DataFrame(0, index=pt_pct.index, columns=self.tickers)
    for idx, val in pt_pct['OFTIC'].items():
      bl_P.loc[idx, val] = 1
      
    self.bl = black_litterman.BlackLittermanModel(pf.cov_matrix, pi=np.array(pf.expected_returns), Q=pt_pct['RET'], P=bl_P, view_confidences=pt_pct['CONFIDENCE'], omega='idzorek')
    return self.bl.bl_returns().to_list()
    