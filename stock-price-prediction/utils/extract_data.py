import yfinance as yf

# Obtain Apple's stock data
ticker_symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2024-01-01'
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Save the data to a CSV file
csv_filename = '../datasets/stock_data.csv'
stock_data.to_csv(csv_filename)
