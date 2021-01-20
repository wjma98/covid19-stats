# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:42:01 2021

@author: William
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt

# Parsing .csv data from NYT to find total deaths. starts on 1/25/20. state-level data
url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv' 
path = 'us-states.csv'

###################################### State-level data 
#Read data
df = pd.read_csv(url, error_bad_lines = False)
df = df.set_index('state')

california = df.loc['California']
california_deaths = df.loc['California', 'deaths']
california_cases = df.loc['California', 'cases']

#compute 1 day lag in order to get daily new cases and deaths
lag_california_deaths =  california_deaths.shift(1)
lag_california_cases = california_cases.shift(1)


#convert series to numpy array
array_deaths = california_deaths.to_numpy() 
array_cases = california_cases.to_numpy()
array_lag_deaths = lag_california_deaths.to_numpy()
array_lag_cases = lag_california_cases.to_numpy()

daily_deaths = array_deaths - array_lag_deaths
daily_cases = array_cases - array_lag_cases

w = np.isnan(daily_deaths)
daily_deaths[w] = 0

w = np.isnan(daily_cases)
daily_cases[w] = 0

#make a smooth-funct. of daily deaths and cases
x2 = np.arange(len(daily_cases)) 

log_daily_deaths = np.where(daily_deaths > 0, np.log(daily_deaths), 0)
log_daily_cases =  np.where(daily_cases > 0, np.log(daily_cases), 0)

s2 = UnivariateSpline(x2, log_daily_deaths, k=3, s=50)
log_best_fit_daily_deaths = s2(x2)
best_fit_daily_deaths = np.exp(log_best_fit_daily_deaths)


s3 = UnivariateSpline(x2, log_daily_cases, k=3, s=25)
log_best_fit_daily_cases = s3(x2)
best_fit_daily_cases = np.exp(log_best_fit_daily_cases)

#compute rolling sums using best_fit
df2 = pd.DataFrame(data = best_fit_daily_cases)
best_fit_roll_12 = df2.rolling(12).sum()
array_best_fit_roll_12 = best_fit_roll_12.to_numpy()

#rolling sums using actual data.
df3 = pd.DataFrame(data = daily_cases)
actual_roll_12 = df3.rolling(12).sum()
array_actual_roll_12 = actual_roll_12.to_numpy()

array_best_fit_roll_12 = np.resize(array_best_fit_roll_12, (len(daily_cases),))
array_actual_roll_12 = np.resize(array_actual_roll_12, (len(daily_cases),))
 
R_e = (best_fit_daily_cases / array_best_fit_roll_12) * 12 #best fit
R_e2 =(daily_cases / array_actual_roll_12) * 12 #actual data

#plotting stuff
figure, axes = plt.subplots(nrows=3, ncols=1, figsize = (13,8), sharex = True)

# =============================================================================
# for the purpose of curve-fitting
# plt.plot(x2, log_daily_deaths, label = 'log daily deaths')
# plt.plot(x2, log_best_fit_daily_deaths, label = 'best fit deaths in log')
# =============================================================================

ones = np.arange(len(x2))
ones.fill(1)

now = dt.datetime.now()
start = now - dt.timedelta(days=len(R_e))
days = mdates.drange(start,now,dt.timedelta(days=1))

axes[0,].scatter(days, daily_cases, color ='red', label = 'data')
axes[0,].plot(days, best_fit_daily_cases, label = 'best fit', color = 'blue')
axes[0,].set(ylabel = 'Daily Cases')

axes[1,].scatter(days, daily_deaths, color ='green', label = 'data')
axes[1,].plot(days, best_fit_daily_deaths, label = 'best fit', color = 'blue')
axes[1,].set(ylabel = 'Daily Deaths')

axes[2,].plot(days, R_e, color = 'orange')
axes[2,].plot(days, ones, '--', color = 'red')
axes[2,].set(ylabel = 'R_0')
axes[2,].set_yticks(np.arange(0.5, 3, 0.5))

plt.subplots_adjust(hspace=0.2)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=30))
plt.gcf().autofmt_xdate()
plt.suptitle("COVID-19 Stats in California")

axes[0,].legend()
axes[1,].legend()

plt.show()

