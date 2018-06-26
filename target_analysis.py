import pandas as pd, numpy as np
import matplotlib.pyplot as plt 

def vintage(start_date, first_date):
    df = pd.DataFrame({'start':start_date, 'first':first_date})
    df['window'] = np.floor(df['first']-df['start'])
    
    freq = pd.pivot_table(df, values='first', index=['start'], columns=['window'], aggfunc=len).fillna(0)
    cum_freq = freq.cumsum(axis=1)
    cum_freq = cum_freq.divide(freq.sum(axis=1), axis=0)
    
    cum_freq.T.plot(kind='line', grid=True).legend(loc='upper left', bbox_to_anchor=(1, 1), title='start') 
    return cum_freq

def delinquency(worst_before,worst_after):
    df = pd.DataFrame({'before':worst_before, 'after':worst_after})
    matrix = pd.pivot_table(df, values='after', index=['before'], columns=['after'], aggfunc=len).fillna(0)
    matrix = matrix.divide(matrix.sum(axis=1), axis=0)
    
    matrix.plot(kind='barh',stacked=True, grid=True, xticks=np.arange(0,1.1,0.1)) \
        .legend(loc='upper left', bbox_to_anchor=(1, 1), title='after') 
    return matrix