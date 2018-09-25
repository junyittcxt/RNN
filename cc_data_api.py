# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 03:36:54 2018

@author: Workstation
"""

import requests
import datetime
import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline
#plt.style.use('fivethirtyeight')

# Pretty print the JSON
import uuid
#from IPython.display import display_javascript, display_html, display
import json

class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json_data
        self.uuid = str(uuid.uuid4())

#    def _ipython_display_(self):
#        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid), raw=True)
#        display_javascript("""
#        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
#        document.getElementById('%s').appendChild(renderjson(%s))
#        });
#        """ % (self.uuid, self.json_str), raw=True)
        
        
def price(symbol, comparison_symbols=['USD'], exchange=''):
    url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}'\
            .format(symbol.upper(), ','.join(comparison_symbols).upper())
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()
    return data

price('LTC', exchange='Coinbase')


def minute_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
    url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}'\
            .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
    if exchange:
        url += '&e={}'.format(exchange)
    page = requests.get(url)
    data = page.json()['Data']
    df = pd.DataFrame(data)
    df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
    return df

time_delta = 1 # Bar width in minutes
df = minute_price_historical('BTC', 'USD', 9999999, time_delta)
df2 = minute_price_historical('LTC', 'USD', 9999, time_delta)
df3= minute_price_historical('BCH', 'USD', 9999, time_delta)
df4 = minute_price_historical('ETH', 'USD', 9999, time_delta)


print('Max length = %s' % len(df))
print('Max time = %s' % (df.timestamp.max() - df.timestamp.min()))

plt.plot(df.timestamp, df.close)
plt.xticks(rotation=45)
plt.show()