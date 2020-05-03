'''TALIB documentation'''

'''MACD'''


macd, macdsignal, macdhist = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)


''' PARABOLIC SAR'''


# Import talib
import talib
# Calculate parabolic sar
data['SAR'] = talib.SAR(data.High, data.Low, acceleration=0.02, maximum=0.2)

# Plot Parabolic SAR with close price
data[['Close', 'SAR']][:500].plot(figsize=(10,5))
plt.grid()
plt.show()

'''ADX'''

real = ADX(high, low, close, timeperiod=14)

'''Bollinger bands'''

up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)


'''RSI'''

 RSI(close, timeperiod=14)
 


''' ATR'''

odayATR = talib.ATR(df1['High'],df1['Low'],df1['Close'],timeperiod=20)

'''Stochastic'''

slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

