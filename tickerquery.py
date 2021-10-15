import requests
import json
import pandas as pd
import numpy as np
import datetime as dt

# Query daily time series for now, may expand this functionality to support diff. types of queries
QUERY = 'TIME_SERIES_DAILY'
REQ_URL = 'https://www.alphavantage.co/query'

# putting this all together, we can finalize the search4ticker function
def search4ticker(ticker: str = 'GOOG', start_date: str = '2019-12-21', end_date: str = '2021-03-19', APIKEY: str = '') -> dict:
    """Return a dict of dates and prices."""
    hist_data = api_query(ticker, QUERY, APIKEY)
    # The returned index is 'Time Series (Daily)'
    hist_df = pd.DataFrame.from_dict(hist_data['Time Series (Daily)'], orient = 'index')
    hist_df['datetimes'] = [dt.datetime.strptime(date, '%Y-%m-%d') for date in hist_df.index]
    # Series of filtering logic
    filtered = hist_df.loc[hist_df.datetimes >= dt.datetime.strptime(start_date, "%Y-%m-%d")]
    filtered = filtered.loc[filtered.datetimes < dt.datetime.strptime(end_date, "%Y-%m-%d")]
    # reverse the dataframe
    filtered = filtered.iloc[::-1]
    # finally, restrict to first 60 rows if the range exceeds it
    filtered = filtered.iloc[range(min(60, filtered.shape[0]))]
    query_result = {}
    query_result['dates'] = np.array(filtered.datetimes, dtype = 'datetime64[D]')
    query_result['prices'] = np.array(filtered['4. close'], dtype = 'float')
    return query_result

# query function
def api_query(ticker, query, api_key):
    params = { 'function'   : QUERY, 
               'outputsize' : 'full', 
               'apikey'     : api_key,
               'datatype'   : 'json',
               'symbol'     : ticker}
    headers = { 'user-agent'   : 'TDI' }

    myResponse = requests.get(REQ_URL, params=params, headers=headers)
    # For successful API call, response code will be 200 (OK)
    if(myResponse.ok):
        # Loading the response data into a dict variable
        # json.loads takes in only binary or string variables so using content to fetch binary content
        # Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
        jData = json.loads(myResponse.content)
        return(jData)
    else:
      # If response code is not ok (200), print the resulting http error code with description
        myResponse.raise_for_status()
        return(None)
