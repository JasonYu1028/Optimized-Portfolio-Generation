from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta

import MVO


def datediff(date1: str, date2: str, interval: str):
  dlta = np.timedelta64(1, 'D')
  if (interval == "d"):
    dlta = np.timedelta64(1, 'D')
  elif (interval == "m"):
    dlta = np.timedelta64(1, 'M')
  elif (interval == "y"):
    dlta = np.timedelta64(1, 'Y')
  else:
    raise NotImplementedError(f"Unknown interval {interval}")
  return (pd.to_datetime(date1) - pd.to_datetime(date2)) / dlta

def dateadd(date: str, interval: str, n: int):
  dlta = relativedelta(days=1)
  if (interval == "d"):
    dlta = relativedelta(days=1)
  elif (interval == "m"):
    dlta = relativedelta(months=1)
  elif (interval == "y"):
    dlta = relativedelta(years=1)
  else:
    raise NotImplementedError(f"Unknown interval {interval}")
  return (datetime.strptime(date, "%Y-%m-%d").date() + dlta * n).strftime("%Y-%m-%d")

def daterange(start_date: str, end_date: str, interval = "d"):
  prev = datetime.strptime(start_date, "%Y-%m-%d").date()
  end = datetime.strptime(end_date, "%Y-%m-%d").date()
  dlta = relativedelta(days=1)
  if (interval == "d"):
    dlta = relativedelta(days=1)
  elif (interval == "m"):
    dlta = relativedelta(months=1)
  elif (interval == "y"):
    dlta = relativedelta(years=1)
  else:
    raise NotImplementedError(f"Unknown interval {interval}")
  while (prev + dlta) <= end:
    yield (prev.strftime("%Y-%m-%d"), (prev + dlta).strftime("%Y-%m-%d"))
    prev += dlta
    
class MVOPlot:
  def __init__(self, portfolio: MVO.Portfolio, reference: MVO.Portfolio, value = 1000) -> None:
    self.portfolio = portfolio.data["Return"].copy()
    self.reference = reference.data["Return"].copy()
    self.value = value
    self.prev = ""
    self.prep_reference()
    
  def prep_reference(self):
    cols = len(self.reference.columns)
    self.reference.iloc[0] = [self.value / cols for _ in range(cols)]

    for idx in self.reference.index[1:]:
      self.reference.loc[idx] = self.reference.shift(1).loc[idx] * (1+self.reference.loc[idx])
      
    self.reference['market'] = self.reference.sum(axis=1)
    
  def add_returns(self, start: str, end: str, weights: list):
    if (self.prev != ""):
      if (datetime.strptime(self.prev, "%Y-%m-%d").date() > datetime.strptime(start, "%Y-%m-%d").date()):
        raise ValueError("add_returns requires that start date is >= previous end date")
    self.prev = end
    curr_pf = self.portfolio[(self.portfolio.index >= start) & (self.portfolio.index < end)]
    self.portfolio.loc[curr_pf.index[0]] = weights * (1+curr_pf.iloc[0]) * self.value
    
    for idx in curr_pf.index[1:]:
      self.portfolio.loc[idx] = self.portfolio.shift(1).loc[idx] * (1+self.portfolio.loc[idx])
    self.value = self.portfolio.loc[curr_pf.index[-1]].sum()
  
  def plot(self):
    sns.lineplot(data=self.reference[["market"]], palette=['y'])
    sns.lineplot(data=pd.DataFrame(self.portfolio.sum(axis=1), columns=["portfolio"]), palette=['b'])