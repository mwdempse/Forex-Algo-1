
#Packages/Moduals
import pandas as pd
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

#ploting

trace0 = go.Ohlc(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name='Currency Quote')
trace1 = go.Scatter(x=df.index,y=ma, line={'color':'blue'})
trace2 = go.Bar(x=df.index,y=df.Volume)

data = [trace0,trace1,trace2]

fig=py.subplots.make_subplots(rows=2,cols=1,shared_xaxes=True)
fig.append_trace(trace0,row=1,col=1)
fig.append_trace(trace1,row=1,col=1)
fig.append_trace(trace2,row=2,col=1)

py.offline.plot(fig,filename='Plot.html')


