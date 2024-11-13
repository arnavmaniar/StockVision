# python3 -m streamlit run main.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from processing import fetch_stock_data, preprocess_data, add_features, train_models, forecast_prices
from visualizations import plot_results, plot_residuals, plot_forecast

#python3 -m streamlit run main.py 
def main():
    st.title('Stock Price Prediction and Forecasting')
    
    tickers = st.text_input('Enter stock tickers (comma-separated)', 'AAPL,MSFT,GOOGL,AMZN,META,TSLA,NFLX')
    tickers = [ticker.strip() for ticker in tickers.split(',')]
    
    start_date = st.date_input('Start Date', value=pd.to_datetime('2016-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2024-09-30'))
    train_end_date = st.date_input('Training End Date', value=pd.to_datetime('2024-03-31'))
    
    if st.button('Run'):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        train_end_date = pd.to_datetime(train_end_date)
        first_ticker = tickers[0]
        st.write(f'Processing {first_ticker}...')
        df = fetch_stock_data(first_ticker, start_date, end_date)
        df = preprocess_data(df)
        df = add_features(df)
        df['Date'] = pd.to_datetime(df['Date'])
        train_df = df[df['Date'] <= train_end_date]
        models = train_models(train_df)
        
        # display MSE and R² values for each model
        st.write("Model Performance on Training Data:")
        for name, result in models.items():
            st.write(f"{name} - MSE: {result['mse']}, R²: {result['r2']}")
        plot_results(train_df, models)
        plot_residuals(models)
        
        # process each ticker for forecasting and plotting
        for ticker in tickers:
            st.write(f'Processing {ticker}...')
            df = fetch_stock_data(ticker, start_date, end_date)
            df = preprocess_data(df)
            df = add_features(df)
            df['Date'] = pd.to_datetime(df['Date'])
            train_df = df[df['Date'] <= train_end_date]
            models = train_models(train_df)
            
            # display MSE and R² values for each model
            st.write(f"Model Performance for {ticker}:")
            for name, result in models.items():
                st.write(f"{name} - MSE: {result['mse']}, R²: {result['r2']}")
            
            # forecast using best
            best_model = models['GradientBoostingRegressor']['model']
            future_df = forecast_prices(df, best_model, start_date=train_end_date + pd.Timedelta(days=1), end_date=end_date)
            
            # plot forecasted w actual prices
            plot_forecast(df, future_df, ticker)

if __name__ == '__main__':
    main()