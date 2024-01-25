#########################################################################################################################
###Owner code: Brent Oeyen
###Comments:
###          -
########################################################################################################################
import os, datetime
import pandas                 as pd
import numpy                  as np
import scipy.stats            as stat
import matplotlib.pyplot      as plt
import matplotlib.dates       as mdates
from scipy.stats              import norm
from pyjstat                  import pyjstat

class data_macro(object):
 
 # Function to transform json object to dataframe
 def json2df(self, link):
  return pyjstat.Dataset.read(link).write('dataframe')
 
 # Query builder to generate data from Eurostat:
 def fetch(self, query, group, name, formatting=0):
  base        = 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/'
  url         = base + query
  df          = self.json2df(url).dropna()
  df.rename(columns={'Geopolitical entity (reporting)':'geo', 'Time frequency': 'Time_frequency', group:'Group'}, inplace=True)
  if formatting==0:
   df.insert(2, 'time', pd.to_datetime(pd.Series(df.Time)), True)
  else:
   df.insert(2, 'time', pd.PeriodIndex(df.Time, freq='Q').to_timestamp(), True)
  df.rename(columns={'Unit of measure': 'Unit'}, inplace=True)
  df.insert(2, name , df.groupby(['geo','Group'])['value'].pct_change(12), True)
  return df
  
 # Determine economic state for each period
 def norm(self, df, name):
  df                   = df[df.notna()]
  df.insert(2, 'Quarter', df.time.map(lambda x: int(x.month/3.1)+1), True)
  df.insert(2, 'Rank'   , df.groupby(['geo','Group'])[name].rank("first", ascending=True), True)
  df                   = pd.merge(df, df.groupby(['geo','Group'], as_index=False)[name].count().rename(columns={name:'N'}), how='inner', on=['geo','Group'], sort=True)
  df.insert(2, 'CDF'    , df.Rank / (df.N + 1), True)
  df.insert(2, 'Z'      , df.CDF.map(lambda x: norm.ppf(x)), True)
  df.insert(2, 'State'  , df.CDF.map(lambda x: 'Bad' if x<0.15 else 'Normal' if x<0.85 else 'Good' if x<1.1 else np.NaN), True)
  df.insert(2, 'Year'   , df.time.map(lambda x: x.year), True)
  df.insert(2, 'Var'    , name, True)
  df.rename(columns={'Value': 'Index', name: 'YoY'}, inplace=True)
  df.drop(columns=['Quarter','Rank'])
  return df
  
if __name__=='__main__':
 #1. Download data
 #1.1 Eurostat
 SIR        = data_macro().fetch('irt_st_m?format=JSON',  'Interest rate', 'IRTst')
 SIR.geo    = SIR.geo.apply(lambda x: 'Euro area' if x=='Euro area (EA11-1999, EA12-2001, EA13-2007, EA15-2008, EA16-2009, EA17-2011, EA18-2014, EA19-2015, EA20-2023)' else x )
 countries  = ['United States', 'Japan', 'United Kingdom', 'Euro area']
 countries2 = ['United Kingdom', 'Euro area']
 CPI        = data_macro().fetch('prc_hicp_midx?format=JSON&precision=8&coicop=CP00&coicop=CP04', 'Classification of individual consumption by purpose (COICOP)', 'CPI').query('Unit=="Index, 2015=100"')
 CPI        = data_macro().norm(CPI[CPI.Group.isin(['All-items HICP'])], 'CPI')
 CPI.geo    = SIR.geo.apply(lambda x: 'Euro area' if x=='Euro area (EA11-1999, EA12-2001, EA13-2007, EA15-2008, EA16-2009, EA17-2011, EA18-2014, EA19-2015, EA20-2023)' else x )
 #1.2 ECB
 ESR         = pd.read_csv('https://data-api.ecb.europa.eu/service/data/EST?format=csvdata')
 ESR.insert(2, 'date', pd.to_datetime(pd.Series(ESR.TIME_PERIOD)), True)
 #2. Plots
 #2.1 interest rates timeseries per country
 plot     = SIR.query('geo == [' + ', '.join(f'"{w}"' for w in countries) + ']')
 plot.insert(2, 't', np.where(plot.Group=='Day-to-day rate', 0, np.where(plot.Group=='1-month rate', 1/12, np.where(plot.Group=='3-month rate', .25, np.where(plot.Group=='6-month rate', .5, 1)))), True)
 fig, ax  = plt.subplots(4, 1)
 for x in range(len(countries)):
  pivot = plot.query('geo == "' + countries[x] + '" and time> "2000-01-01"').pivot_table(values='value', columns='Group', index='time')
  ax[x].plot(pivot)
  ax[x].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
  ax[x].set_title(countries[x])
  ax[x].legend(pivot.columns, loc='lower left')
 plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=.4, hspace=.4)
 plt.show()
 #2.2 Yield curve Europe
 fig, ax = plt.subplots(1, 2)
 for x in range(len(countries2)):
  pivot = plot.query('geo == "' + countries2[x] + '" and time> "2021-12-31"').where(plot.time.dt.month.isin([1, 6, 12])).pivot_table(values='value', columns='time', index='t')
  ax[x].plot(pivot)
  ax[x].set_title(countries2[x])
  ax[x].legend(pivot.columns, loc='lower left')
 plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=.4, hspace=.4)
 plt.show()
 #2.3 Euro Short Rate
 plot2   = ESR.query('TITLE_COMPL=="Euro short-term rate - Volume-weighted trimmed mean rate - Unsecured - Overnight - Borrowing - Financial corporations"')
 fig, ax = plt.subplots()
 ax.xaxis.set_major_locator(mdates.YearLocator())
 ax.xaxis.set_minor_locator(mdates.MonthLocator())
 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 ax.plot(plot2.date, plot2.OBS_VALUE)
 ax.set_title(' euro short-term rate (STR) timeseries'); fig.autofmt_xdate()
 plt.show()
 #2.4 Inflation timeseries per country
 plot3    = CPI.query('geo == [' + ', '.join(f'"{w}"' for w in countries) + ']')
 fig, ax  = plt.subplots()
 pivot    = plot3.query('Year>2006').pivot_table(values='YoY', columns='geo', index='time')
 ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 ax.plot(pivot); ax.set_title('Inflation timeseries'); ax.legend(pivot.columns, loc='lower left')
 plt.subplots_adjust(left=.1, bottom=.1, right=.9, top=.9, wspace=.4, hspace=.4)
 plt.show()
