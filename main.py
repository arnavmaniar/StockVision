import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from processing import fetch_stock_data, preprocess_data, add_features, train_models, forecast_prices
from visualizations import plot_results, plot_residuals, plot_forecast

def main():
    tickers = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'META': 'Meta',
        'TSLA': 'Tesla',
        'NFLX': 'Netflix'
    }
    start_date = '2016-01-01'
    end_date = '2024-09-30'
    train_end_date = '2024-03-31'
    
    # Process the first ticker to print training and residuals
    first_ticker = list(tickers.keys())[0]
    print(f'Processing {first_ticker}...')
    
    # Fetch data
    df = fetch_stock_data(first_ticker, start_date, end_date)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Add features
    df = add_features(df)
    
    # Limit training data to March 31st, 2024
    train_df = df[df['Date'] <= train_end_date]
    
    # Train models
    models = train_models(train_df)
    
    # Plot results
    plot_results(train_df, models)
    
    # Plot residuals
    plot_residuals(models)
    
    # Process each ticker for forecasting and plotting
    for ticker, company_name in tickers.items():
        print(f'Processing {ticker}...')
        
        # Fetch data
        df = fetch_stock_data(ticker, start_date, end_date)
        
        # Preprocess data
        df = preprocess_data(df)
        
        # Add features
        df = add_features(df)
        
        # Limit training data to March 31st, 2024
        train_df = df[df['Date'] <= train_end_date]
        
        # Train models
        models = train_models(train_df)
        
        # Forecast future prices using the best model (e.g., GradientBoostingRegressor)
        best_model = models['GradientBoostingRegressor']['model']
        future_df = forecast_prices(df, best_model, start_date='2024-04-01', end_date='2024-09-30')
        
        # Plot forecasted prices with actual prices
        plot_forecast(df, future_df, company_name)

if __name__ == '__main__':
    main()