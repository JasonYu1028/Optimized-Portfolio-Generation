from typing import Union

import numpy as np
import pandas as pd


class Analyst:
  """Analyst Class - Retrieve confidence levels about every analyst"""
  def __init__(self, ratings: pd.DataFrame, historical_data: pd.DataFrame) -> None:
    # Get ratings data
    self.ratings = ratings[['OFTIC', 'ANNDATS', 'AMASKCD', 'IRECCD']]
    # Disregard NEUTRAL ratings
    self.ratings = self.ratings[self.ratings['IRECCD'] != '3']
    self.ratings['IRECCD'] = self.ratings['IRECCD'] > "3"
    self.ratings.columns = ['OFTIC', 'DATE', 'AMASKCD', 'IRECCD']
  
    # Get full historical data (Large File)
    self.hist_data = historical_data[['DATE', 'TICKER', 'PRC']]
    self.hist_data = self.hist_data.dropna()
    self.hist_data = self.hist_data.drop_duplicates(subset=['DATE', 'TICKER'])
    # Process the data to prepare for merge
    self.hist_data = self.hist_data.pivot(columns='TICKER', index='DATE', values='PRC')
    self.hist_data = self.hist_data.resample('1d').mean().ffill()
    
    # Preprocess historical data to identify growth or decline (monthly)
    self.bool_growth = (self.hist_data - self.hist_data.shift(periods=-30)).applymap(lambda x: np.NaN if np.isnan(x) else x > 0)
    self.bool_growth = self.bool_growth.melt(var_name='OFTIC',value_name='IS_GROW', ignore_index=False)
    self.bool_growth.dropna(inplace=True)
    self.bool_growth.reset_index(names=['DATE'], inplace=True)
    
    # Merge the two datasets
    self.merged = pd.merge(self.ratings, self.bool_growth, on=['OFTIC', 'DATE'], how='inner')
  
  def rank(self, end_date: Union[str, None] = None) -> pd.DataFrame:
    """Get confidence levels without time bias - confidence prior to end_date"""
    data = self.merged
    if end_date is not None:
      data = self.merged[self.merged['DATE'] <= end_date]
    grouped = data.groupby('AMASKCD')
    analysts = {'AMASKCD': [], 'CONFIDENCE': []}
    for name, group in grouped:
      analysts['AMASKCD'].append(name)
      conf = (group['IRECCD'] == group['IS_GROW']).sum() / len(group)
      if len(group) <= 1:
        conf = 0.5
      analysts['CONFIDENCE'].append(conf)
    self.confidence = pd.DataFrame(analysts)