#Packages/Moduals
import pandas as pd
from algo_functions import *
import plotly as py
from plotly import tools as tls
import plotly.graph_objs as go

#import/clean data
df=pd.read_csv('AUDCADH1.csv')
df.Time=pd.to_datetime(df.Time,format='%Y.%m.%d %H:%M:%S')
df=df.set_index(df.Time)
df=df[['Open','High','Low','Close','Volume']]
print(df.head()) 
df = df.iloc[:200]

#SMA
ma = df.Close.rolling(center=False,window=30).mean()


# Fourier 

f = fourier(df,[10,15],method='difference',to_plot=True)
s= sine(df,[10,15],method='difference',to_plot=True)

