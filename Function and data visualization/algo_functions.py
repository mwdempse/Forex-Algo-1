#Packages/Moduals
import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
# import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from matplotlib.finance import candlestick
# from matplotlib.dates import date2num
# from datetime import datetime


class holder: 
    1
#heiken ashi candles

def heikenashi(prices,periods):
    
    
    """
    
    :param prices: dataframe of OHLC with Volume data
    :param periods: periods for which to make the candles
    :return: heiken ashi OHLC candles
    
    """
    
    results = holder()
    
    dict = {}
    
    HAclose = prices.iloc[:,0:4].sum(axis=1)/4
    
    HAopen = HAclose.copy()
    
    HAopen.iloc[0] = HAclose.iloc[0]
    
    HAhigh = HAclose.copy()
    
    HAlow =  HAclose.copy()
    
    for i in range(1,len(prices)):
        
        HAopen.iloc[i] = (HAopen.iloc[i-1] + HAclose.iloc[i-1])/2
        HAhigh.iloc[i] = np.array([prices.High.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).max()
        HAlow.iloc[i] = np.array([prices.Low.iloc[i],HAopen.iloc[i],HAclose.iloc[i]]).min()
        
    df = pd.concat((HAopen,HAhigh,HAlow,HAclose),axis=1)
    df.columns = ['open','high','low','close']
    
    #df.index = df.index.droplevel(0)
    
    dict[periods[0]]=df
    
    results.candles = dict
    
    return results

#Detrend
    
def detrend(prices, method='difference'):
    
    """
    
    :param prices: dataframe of OHLC
    :param method: method for detrending 'linear' or 'difference'
    :return: the detrended data
    
    """
    
    if method == 'difference':
        
        detrended = prices.Close[1:]-prices.Close[:-1].values
    
    elif method == 'linear':
        x=np.arange(0,len(prices))
        y=prices.Close.values
        
        model=LinearRegression()
        
        model.fit(x.reshape(-1,1),y.reshape(-1,1))
        
        trend = model.predict(x.reshape(-1,1))
        
        trend=trend.reshape((len(prices),))
        
        detrended = prices.Close-trend
        
    else:
        
        print("Missing 'linear' or 'difference' as method")
        
    return detrended

#Fourier Expantion 

def fseries(x,a0,a1,b1,w):
    
    """
    :param x: hours
    :param a0: first fourier coefficient
    :param a1: second fourier coefficient
    :param b1: third fourier coefficient
    :param w: fourier frequency
    :return: value of the fourier function
    
    """
    
    f = a0 + a1*np.cos(w*x) + b1 * np.sin(w*x)
    
    return f

#Sine Expantion

def sseries(x,a0,b1,w):
    
    """
    :param x: hours
    :param a0: first sine coefficient
    :param b1: second sine coefficient
    :param w: sine frequency
    :return: value of the sine function
    
    """
    
    s = a0 + b1 * np.sin(w*x)
    
    return s

#Fourier Series Coefficient

def fourier(prices, periods, method='difference', to_plot=False):

    '''
    param price:      OHLC dataframe
    param periods:    list of periods to compute coefficients
    param method:     method for detrend
    return:           dict of df containing coefficients for said periods
    '''

    results = holder()
    dict = {}

    # compute the coefficients of the series

    detrended = detrend(prices, method)
    
    for i in range(0, len(periods)):

        coeffs = []

        for j in range(periods[i], len(prices)-periods[i]):

            x = np.arange(0, periods[i])
            y = detrended.iloc[j-periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fseries, x, y)
                except (RuntimeError, OptimizeWarning):
                    res = np.empty((1,4))
                    res[0,:] = np.nan
                
            if to_plot == True:

                xt = np.linspace(0, periods[i], 100)
                yt = fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                
                plt.show()
            
            coeffs.extend([res[0]])

        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        
        coeffs = np.array(coeffs)
        
        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]].index)

        df.columns = [['a0','a1','b1','w']]

        df = df.fillna(method='bfill')

        dict[periods[i]] = df

    results.coeffs = dict

    return results


#Sine Series Coefficient

def sine(prices, periods, method='difference', to_plot=False):

    '''
    param price:      OHLC dataframe
    param periods:    list of periods to compute coefficients
    param method:     method for detrend
    return:           dict of df containing coefficients for said periods
    '''

    results = holder()
    dict = {}

    # compute the coefficients of the series

    detrended = detrend(prices, method)
    
    for i in range(0, len(periods)):

        coeffs = []

        for j in range(periods[i], len(prices)-periods[i]):

            x = np.arange(0, periods[i])
            y = detrended.iloc[j-periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error',OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fseries, x, y)
                except (RuntimeError, OptimizeWarning):
                    res = np.empty((1,3))
                    res[0,:] = np.nan
                
            if to_plot == True:

                xt = np.linspace(0, periods[i], 100)
                yt = fseries(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x,y)
                plt.plot(xt,yt,'r')
                
                plt.show()
            
            coeffs.extend([res[0]])

        warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
        
        coeffs = np.array(coeffs)
        
        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]].index)

        df.columns = [['a0','b1','w']]

        df = df.fillna(method='bfill')

        dict[periods[i]] = df

    results.coeffs = dict

    return results


#Williams Accumulation Distribution
    
def wadl(prices,periods):
    
    '''
    param price:      OHLC dataframe
    param periods:    list of periods to compute coefficients
    return:           Williams Accumulation distribution
    '''
    
    results = holder()
    dict = {}
    
    for i in range(0, len(periods)):
        WAD = []
        
        for j in range(periods[i],len(prices)-periods[i]):
            TRH = np.array([prices.High.iloc[j],prices.Close.iloc[j-i]]).max()
            TRL = np.array([prices.Low.iloc[j],prices.Close.iloc[j-i]]).min()
            
            if prices.Close.iloc[j] > prices.Close.iloc[j-1]:
                PM = prices.Close.iloc[j] - TRL
                
            elif prices.Close.iloc[j] < prices.Close.iloc[j-1]:
                PM = prices.Close.iloc[j] - TRH
                
            elif prices.Close.iloc[j] == prices.Close.iloc[j-1]:
                PM = 0
            
            else:
                
                print('clean data better')
            
            AD=PM * prices.Volume.iloc[j]
            WAD = np.append(WAD, AD)
            
        WAD = WAD.cumsum()
        
        WAD = pd.DataFrame(WAD,index=prices.iloc[periods[i]:-periods[i]].index)
        
        WAD.columns = ['Close']
        
        dict[periods[i]] = WAD
        
    results.wadl = dict
        
    return results
            
#TimeFrame Resampler
def OHLCresample(DataFrame,TimeFrame,column='ask'):
    
    '''
    param DataFrame:  OHLC dataframe with data to be resampled
    param TimeFrame:  timeframe for resampling
    param column:     which column to resample (bid or ask) default = 'ask'
    return:           resampled OHLC data for the given timeframe
    
    '''

    grouped  = DataFrame.groupby('Symbol')
    
    if np.any(DataFrame.columns=='Ask'):
        if column == 'ask':
            ask = grouped['Ask'].resample(TimeFrame).ohlc()
            askVol = grouped['Volume'].resample(TimeFrame).count()
            resampled = pd.DataFrame(ask)
            resampled['Volume'] = askVol
        
        elif column == 'bid':
            bid = grouped['Bid'].resample(TimeFrame).ohlc()
            bidVol = grouped['Volume'].resample(TimeFrame).count()
            resampled = pd.DataFrame(ask)
            resampled['Volume'] = bidVol
    
        else:
        
            raise ValueError('Column but be a string. Either ask or bid')
    
    elif np.any(DataFrame.columns=='Close'):
        open = grouped['Open'].resample(TimeFrame).ohlc()
        high = grouped['High'].resample(TimeFrame).ohlc()
        low = grouped['Low'].resample(TimeFrame).ohlc()
        close = grouped['Close'].resample(TimeFrame).ohlc()
        askVol = grouped['Volume'].resample(TimeFrame).count()
        
        resampled=pd.DataFrame(open)
        resampled['High'] = high
        resampled['Low'] = low
        resampled['Close'] = close
        resampled['Volume'] = askVol
        
    resampled = resampled.dropna()
    
    return resampled

def momentum(prices, periods):
    
    '''
    param prices:  OHLC dataframe
    param periods:  list of periods to calculate value
    param column:     which column to resample (bid or ask) default = 'ask'
    return:           momemtum indicator
    
    '''
    
    results = holder()
    open = {}
    close= {}
    
    for i in range(0,len(prices)):
        open[periods[i]] = pd.DataFrame(list(zip(prices.Open.iloc[periods[i]:]-prices.Open.iloc[:-periods[i]].values)),
                                        index=prices.iloc[periods[i]:].index,
                                        columns=['Open'])
        
        close[periods[i]] = pd.DataFrame(list(zip(prices.Close.iloc[periods[i]:]-prices.Close.iloc[:-periods[i]].values)),
                                        index=prices.iloc[periods[i]:].index,
                                        columns=['Close'])
        
        results.Open = open
        results.Close= close
        
        return results
    
# Stochastic Oscillator

def stochastic(prices, periods):
    
    '''
    param prices:  OHLC dataframe
    param periods:  list of periods to calculate value
    return:         Stochastic oscillator indicator values
    
    '''
    
    results = holder()
    close  ={}
    
    for i in range(0,len(prices)):
        Ks = []
        
        for j in range(periods[i],len(prices)-periods[i]):
            
            H=prices.High.iloc[j-periods[i]:j].max()
            L=prices.Low.iloc[j-periods[i]:j].min()
            C=prices.Close.iloc[j+1]
            
            if H==L:
                K=0
            else:
                K=100*(C-L)/(H-L)
                
            Ks = np.append(Ks,K)
            
            df = pd.DataFrame(Ks, index=prices.iloc[periods[i]+1:-periods[i]+1].index)
            df.columns = [['K']]
            df['D'] = df.K.rolling(3).mean()
            
            close[periods[i]] = df
            
        results.Close = close
        
        return results
    
# Williams %R

def williams(prices, periods):
    
    '''
    param prices:  OHLC dataframe
    param periods:  list of periods to calculate value
    return:         Williams %R indicator values
    
    '''
    
    results = holder()
    close  ={}
    
    for i in range(0,len(prices)):
        Rs = []
        
        for j in range(periods[i],len(prices)-periods[i]):
            
            H=prices.High.iloc[j-periods[i]:j].max()
            L=prices.Low.iloc[j-periods[i]:j].min()
            C=prices.Close.iloc[j+1]
            
            if H==L:
                R=0
            else:
                R=100*(H-C)/(H-L)
                
            Rs = np.append(Rs,R)
            
            df = pd.DataFrame(Rs, index=prices.iloc[periods[i]+1:-periods[i]+1].index)
            df.columns = [['R']]
            df['D'] = df.K.rolling(3).mean()
            
            close[periods[i]] = df
            
        results.Close = close
        
        return results
    
#Price rate of change

def proc(prices, periods):
    
    '''
    param prices:  OHLC dataframe
    param periods:  list of periods to calculate value
    return:         Price rate of change indicator values
    
    '''
    
    results = holder()
    proc = {}
    
    for i in 0,len(prices)):
    
        proc[periods[i]] = pd.DataFrame((prices.Close.iloc[periods[i]:]-prices.Close.iloc[:-periods[i]].values)\
                                        /prices.close.iloc[:-periods[i]].values)
            
        proc[periods[i]].columns = [['Close']]
        
        results.proc = proc
        
        return results
    
# Accumulation Distribution Oscillator

def adosc(prices, periods):
    
    '''
    param prices:  OHLC dataframe
    param periods:  list of periods to calculate value
    return:         Accumulation Distribution Oscillator indicator values
    
    '''
    
    results = holder()
    accdist  ={}
    
    for i in range(0,len(prices)):
        AD = []
        
        for j in range(periods[i],len(prices)-periods[i]):
            
            H=prices.High.iloc[j-periods[i]:j].max()
            L=prices.Low.iloc[j-periods[i]:j].min()
            C=prices.Close.iloc[j+1]
            V=prices.AskVol.iloc[j+1]
            
            if H==L:
                CLV=0
            else:
                CLV=100*((C-L)-(H-C))/(H-L)
                
            AD = np.append(AD,CLV*V)
            
            AD = AD.cumsum()
            
            AD = pd.DataFrame(AD, index=prices.iloc[periods[i]+1:-periods[i]+1].index)
            AD.columns = [['AD']]
            
            accdist[periods[i]] = AD
            
        results.AD = accdist
        
        return results
