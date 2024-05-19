from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn import covariance, linear_model


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
    return pd.DataFrame(analysts)

class Market:
  """Market Class (Abstract)
  
  Params:
   - start: start date to retrieve historical data
   - end: end date to retrieve historical data
  """
  def __init__(self, start: str, end: Union[str, None] = None) -> None:
    self.start = start
    self.end = end
  
  def retrieve_data(self):
    raise NotImplementedError("Data retrieval not implemented")
    
  def calc_stats(self):
    self.calc_risk_free_rate()
    self.calc_volatility()
    
  def calc_risk_free_rate(self):
    raise NotImplementedError("Rf rate not implemented")
    
  def calc_volatility(self):
    raise NotImplementedError("Market volatility not implemented")

  def __str__(self) -> str:
    return f'Market Reference: {self.ticker}\n  Risk-free rate: {self.risk_free_rate}\n  True Volatility: {self.volatility}'
  
class MarketIndex(Market):
  """Index-based Market Class
  
  Params:
   - ticker: index to track
   - start: start date to retrieve historical data
   - end: end date to retrieve historical data
  """
  def __init__(self, data: pd.DataFrame,  start: str, end: Union[str, None] = None) -> None:
    self.data = data
    Market.__init__(self, start, end)
    self.retrieve_data()
    self.calc_stats()
    
  def retrieve_data(self):
    if self.end is None:
      self.data = self.data[(self.data.index >= self.start)]
    else:
      self.data = self.data[(self.data.index >= self.start) & (self.data.index < self.end)]
    self.data = self.data[['Close']]
    self.data.dropna(inplace=True)
    self.data['Return'] = self.data['Close'].pct_change()
    
  def calc_risk_free_rate(self):
    self.risk_free_rate = np.mean(self.data['Return'])
    
  def calc_volatility(self):
    self.volatility = np.std(self.data['Return'])
    
class MarketFF(Market):
  """Fama-French Data-based Market Class
  
  Params:
   - file: csv file from Ken French's Fama-French database
   - start: start date to retrieve historical data
   - end: end date to retrieve historical data
  """
  def __init__(self, data: pd.DataFrame, start: str, end: Union[str, None] = None) -> None:
    self.data = data
    Market.__init__(self, start, end)
    self.retrieve_data()
    self.calc_stats()
    
  def retrieve_data(self):
    self.data.columns = ['DATE', 'MKTRF', 'SMB', 'HML', 'RF']
    self.data['RF'] = self.data['RF'] / 100
    self.data['DATE'] = pd.to_datetime(self.data['DATE'], format='%Y%m%d')
    if self.end is None:
      self.data = self.data[(self.data['DATE'] >= self.start)]
    else:
      self.data = self.data[(self.data['DATE'] >= self.start) & (self.data['DATE'] <= self.end)]
      
  def calc_risk_free_rate(self):
    self.risk_free_rate = np.mean(self.data['RF'])
    
  def calc_volatility(self):
    self.volatility = 0
    
class Portfolio:
  """Portfolio Class
  
  Params:
   - tickers: the list of stocks to generate the portfolio
   - start: start date to retrieve historical data
   - end: end date to retrieve historical data
  """
  def __init__(self, data: pd.DataFrame, start: str, end: Union[str, None] = None) -> None:
    self.data = data
    self.start = start
    self.end = end
    self.tickers = data['Close'].columns
    self.retrieve_data()
    self.calc_stats()
    
  def __str__(self) -> str:
    return f'Portfolio Assets: {self.tickers}\n  Average Return: {self.mean_returns}\n  Expected Return: {self.expected_returns}'
    
  def retrieve_data(self):
    self.data = self.data[['Close']]
    if self.end is None:
      self.data = self.data[(self.data.index >= self.start)]
    else:
      self.data = self.data[(self.data.index >= self.start) & (self.data.index <= self.end)]
    
    returns = self.data[['Close']].pct_change()
    returns.columns = returns.columns.set_levels(['Return'], level=0)
    self.data = pd.concat([self.data, returns], axis=1)
  
  def calc_stats(self):
    self.calc_mean_returns()
    self.calc_expected_returns()
    self.calc_covariance_matrix()
  
  def calc_mean_returns(self):
    self.mean_returns = [np.mean(self.data['Return'][ticker]) for ticker in self.tickers]
    
  def calc_expected_returns(self):
    """Uses the Fama-French 3 Factor Model"""
    returns = self.data['Return'].reset_index().melt(id_vars=["Date"], var_name="ASSET", value_name="RET").dropna()
    returns.columns = ['DATE', 'ASSET', 'RET']
    
    ff3 = pd.read_csv('ff_daily.csv')
    ff3.columns = ['DATE', 'MKTRF', 'SMB', 'HML', 'RF']
    ff3['RF'] = ff3['RF'] / 100
    ff3['DATE'] = pd.to_datetime(ff3['DATE'], format='%Y%m%d')
    
    merged = pd.merge(returns, ff3, on="DATE")
    merged["XRET"] = merged["RET"] - merged["RF"]
    grouped = merged.groupby(["ASSET"])

    beta = {'ASSET':[], 'ff3_alpha':[], 'ff3_beta':[], 'smb_beta':[], 'hml_beta':[]}
    ret = {'ASSET': [], 'DATE': [], 'FF3_RET': []}

    for name, group in grouped:
        ff3model = linear_model.LinearRegression().fit(group[["MKTRF", "SMB", "HML"]], group["XRET"])
        
        beta['ASSET'].append(name)
        beta['ff3_alpha'].append(ff3model.intercept_)
        beta['ff3_beta'].append(ff3model.coef_[0])
        beta['smb_beta'].append(ff3model.coef_[1])
        beta['hml_beta'].append(ff3model.coef_[2])
        
        ret['ASSET'].extend([name] * len(group))
        ret['DATE'].extend(group["DATE"])
        ret['FF3_RET'].extend(ff3model.predict(group[["MKTRF", "SMB", "HML"]]) + group["RF"])
        
    self.beta = pd.DataFrame(beta)
    self.beta['ERET'] = self.beta['ff3_alpha'] + self.beta['ff3_beta'] * np.mean(merged['MKTRF']) + self.beta['smb_beta'] * np.mean(merged['SMB']) + self.beta['hml_beta'] * np.mean(merged['HML'])
    self.expected_returns = self.beta['ERET'].to_list()
    
  def calc_covariance_matrix(self):
    """Calculates the covariance matrix using OAS shrinkage"""
    # Use weekly data for better covariance results
    weekly_returns = self.data['Return'].resample('1W').mean().applymap(lambda x: (x + 1)**7 - 1).fillna(0)
    self.cov_matrix = covariance.oas(weekly_returns)[0]
    
class MVO:
  """Mean Variance Optimization Class
  
  Params:
   - portfolio: the portfolio to optimize
   - market: the market reference
   - bounds: constraint on the weight of each stock where 1 == 100% and negative means shorting
  """
  def __init__(self, portfolio: Portfolio, market: Market, bounds: Union[list, None] = None) -> None:
    self.portfolio = portfolio
    self.market = market
    self.bounds = bounds
  
  def pf_variance(self, weights):
    return np.dot(weights.T, np.dot(self.portfolio.cov_matrix, weights))
  
  def pf_volatility(self, weights):
    return np.sqrt(self.pf_variance(weights))
  
  def pf_return(self, weights):
    return np.dot(weights.T, self.portfolio.expected_returns)
  
  def pf_return_negate(self, weights):
    return -np.dot(weights.T, self.portfolio.expected_returns)
  
  def pf_sharpe(self, weights, risk_free_rate):
    return -(self.pf_return(weights) - risk_free_rate) / self.pf_volatility(weights)
    
  def pf_utility(self, weights, risk_aversion):
    return -(self.pf_return(weights) - (1 / 2) * risk_aversion * self.pf_volatility(weights))
  
  def minimum_variance(self, expected_return=None):
    init_guess = np.ones(len(self.portfolio.tickers)) / len(self.portfolio.tickers)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if not (expected_return is None):
      constraints.append({"type": "eq", "fun": lambda x: self.pf_return(x) - expected_return})

    result = minimize(fun=self.pf_variance,
                      x0=init_guess,
                      args=(),
                      method="SLSQP",
                      constraints=constraints,
                      bounds=self.bounds)

    self.stats = {
      "Weights": result.x,
      "Returns": np.sum(self.portfolio.expected_returns * result.x),
      "Volatility": np.sqrt(result.fun)
    }
    
  def maximum_return(self, expected_volatility):
    init_guess = np.ones(len(self.portfolio.tickers)) / len(self.portfolio.tickers)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
                   {"type": "eq", "fun": lambda x: self.pf_volatility(x) - expected_volatility}]

    result = minimize(fun=self.pf_return_negate,
                      x0=init_guess,
                      args=(),
                      method="SLSQP",
                      constraints=constraints,
                      bounds=self.bounds)

    self.stats = {
      "Weights": result.x,
      "Returns": -result.fun,
      "Volatility": self.pf_volatility(result.x)
    }
  
  def maximum_sharpe(self):
    init_guess = np.ones(len(self.portfolio.tickers)) / len(self.portfolio.tickers)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

    result = minimize(fun=self.pf_sharpe,
                      x0=init_guess,
                      args=(self.market.risk_free_rate),
                      method="SLSQP",
                      constraints=constraints,
                      bounds=self.bounds)

    self.stats = {
      "Weights": result.x,
      "Returns": np.sum(self.portfolio.expected_returns * result.x),
      "Volatility": self.pf_volatility(result.x),
      "Sharpe": -result.fun
    }
    
  def maximum_utility(self, risk_aversion=10):
    init_guess = np.ones(len(self.portfolio.tickers)) / len(self.portfolio.tickers)    

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

    result = minimize(fun=self.pf_utility,
                      x0=init_guess,
                      args=(risk_aversion),
                      method="SLSQP",
                      constraints=constraints,
                      bounds=self.bounds)

    self.stats = {
      "Weights": result.x,
      "Returns": np.sum(self.portfolio.expected_returns * result.x),
      "Volatility": self.pf_volatility(result.x),
      "Utility": -result.fun
    }
    
  def plot_MVO(self):
    self.minimum_variance()
    returns = np.linspace(self.stats['Returns'] - 0.001, self.stats['Returns'] + 0.002, 51)
    vol = []
    for ret in returns:
      self.minimum_variance(ret)
      vol.append(self.stats["Volatility"])
    plt.plot(vol, returns)
    plt.title('Mean Variance Frontier')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()
