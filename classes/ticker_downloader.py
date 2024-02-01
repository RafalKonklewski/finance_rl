import yfinance as yf


class TickerDownloader(object):
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        try:
            df = yf.download(self.tickers, self.start_date, self.end_date)
            return df
        except Exception as e:
            print(e)
            return None
        
#Test download_data
