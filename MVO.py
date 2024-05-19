from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn import covariance, linear_model


class Market:
  """Fama-French Data-based Market Class
  
  Params:
   - data: Fama French data from Ken French's Fama-French database
   - start: start date to retrieve historical data
   - end: end date to retrieve historical data
   
   Data Format: 'DATE' | 'MKTRF' | 'SMB' | 'HML' | 'RF'
  """
  def __init__(self, data: pd.DataFrame, start: str, end: Union[str, None] = None) -> None:
    self.data = data
    self.start = start
    self.end = end
    self.process_data()
    self.calc_stats()
    
  def process_data(self):
    self.data.columns = ['DATE', 'MKTRF', 'SMB', 'HML', 'RF']
    self.data['RF'] = self.data['RF'] / 100
    self.data['DATE'] = pd.to_datetime(self.data['DATE'], format='%Y%m%d')
    if self.end is None:
      self.data = self.data[(self.data['DATE'] >= self.start)]
    else:
      self.data = self.data[(self.data['DATE'] >= self.start) & (self.data['DATE'] <= self.end)]
      
  def calc_stats(self):
    self.calc_risk_free_rate()
    
  def calc_risk_free_rate(self):
    self.risk_free_rate = np.mean(self.data['RF'])
    
class Portfolio:
  """Portfolio Class
  
  Params:
   - tickers: the list of stocks to generate the portfolio
   - start: start date to retrieve historical data
   - end: end date to retrieve historical data
  """
  def __init__(self, data: pd.DataFrame, ff_data: pd.DataFrame, start: str, end: Union[str, None] = None) -> None:
    self.data = data
    if (self.data.columns.nlevels == 1):
      self.data.columns = pd.MultiIndex.from_product([self.data.columns] + [["TICKER"]])
    self.ff_data = ff_data
    self.start = start
    self.end = end
    self.tickers = data['Close'].columns.tolist()
    self.tickers.sort()
    self.process_data()
    self.calc_stats()
    
  def __str__(self) -> str:
    return f"""Portfolio Assets: {self.tickers}\n
               Average Return:   {self.mean_returns}\n
               Expected Return:  {self.expected_returns}"""
    
  def process_data(self):
    self.data = self.data[['Close']]
    if self.end is None:
      self.data = self.data[(self.data.index >= self.start)]
    else:
      self.data = self.data[(self.data.index >= self.start) & (self.data.index <= self.end)]
    
    returns = self.data[['Close']].pct_change().fillna(0)
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
    
    ff3 = Market(self.ff_data, self.start, self.end).data
    
    merged = pd.merge(returns, ff3, on="DATE")
    merged["XRET"] = merged["RET"] - merged["RF"]
    grouped = merged.groupby("ASSET")

    beta = {'ASSET':[], 'ff3_alpha':[], 'ff3_beta':[], 'smb_beta':[], 'hml_beta':[]}

    for name, group in grouped:
      ff3model = linear_model.LinearRegression().fit(group[["MKTRF", "SMB", "HML"]], group["XRET"])
      
      beta['ASSET'].append(name)
      beta['ff3_alpha'].append(ff3model.intercept_)
      beta['ff3_beta'].append(ff3model.coef_[0])
      beta['smb_beta'].append(ff3model.coef_[1])
      beta['hml_beta'].append(ff3model.coef_[2])
        
    self.beta = pd.DataFrame(beta)
    self.beta['ERET'] = (self.beta['ff3_alpha'] + self.beta['ff3_beta'] * np.mean(merged['MKTRF'])
                                                + self.beta['smb_beta'] * np.mean(merged['SMB']) 
                                                + self.beta['hml_beta'] * np.mean(merged['HML']))
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
  def __init__(self, portfolio: Portfolio, bounds: Union[list, None] = None) -> None:
    self.portfolio = portfolio
    self.bounds = bounds
    self.init_guess = np.ones(len(self.portfolio.tickers)) / len(self.portfolio.tickers)
  
  def pf_variance(self, weights):
    return np.dot(weights.T, np.dot(self.portfolio.cov_matrix, weights))
  
  def pf_volatility(self, weights):
    return np.sqrt(self.pf_variance(weights))
  
  def pf_return(self, weights):
    return np.dot(weights.T, self.portfolio.expected_returns)
  
  def pf_return_negate(self, weights):
    return -self.pf_return(self, weights)
  
  def pf_sharpe(self, weights, risk_free_rate):
    return -(self.pf_return(weights) - risk_free_rate) / self.pf_volatility(weights)
    
  def pf_utility(self, weights, risk_aversion):
    return -(self.pf_return(weights) - (1 / 2) * risk_aversion * self.pf_volatility(weights))
  
  def minimum_variance(self, expected_return=None):
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    if expected_return:
      constraints.append({"type": "eq", "fun": lambda x: self.pf_return(x) - expected_return})

    result = minimize(fun=self.pf_variance,
                      x0=self.init_guess,
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
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
                   {"type": "eq", "fun": lambda x: self.pf_volatility(x) - expected_volatility}]

    result = minimize(fun=self.pf_return_negate,
                      x0=self.init_guess,
                      args=(),
                      method="SLSQP",
                      constraints=constraints,
                      bounds=self.bounds)

    self.stats = {
      "Weights": result.x,
      "Returns": -result.fun,
      "Volatility": self.pf_volatility(result.x)
    }
  
  def maximum_sharpe(self, risk_free_rate):
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

    result = minimize(fun=self.pf_sharpe,
                      x0=self.init_guess,
                      args=(risk_free_rate),
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
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

    result = minimize(fun=self.pf_utility,
                      x0=self.init_guess,
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
