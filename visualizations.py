import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Plot results
def plot_results(df: pd.DataFrame, models: dict):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Actual Prices', color='black')
    
    for name, model_info in models.items():
        model = model_info['model']
        predictions = model.predict(df[['Year', 'Month', 'Day', 'DayOfWeek', 'IsMonthStart', 'IsMonthEnd', 'Lag1', 'Lag2', 'Lag3']])
        plt.plot(df['Date'], predictions, label=f'{name} Predictions')
    
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Predictions')
    plt.legend()
    plt.show()

# Plot residuals
def plot_residuals(models: dict):
    plt.figure(figsize=(14, 7))
    
    for name, model_info in models.items():
        residuals = model_info['y_test'] - model_info['predictions']
        sns.histplot(residuals, kde=True, label=f'{name} Residuals')
    
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.legend()
    plt.show()

# Plot forecasted prices with actual prices
def plot_forecast(df: pd.DataFrame, future_df: pd.DataFrame, company_name: str):
    plt.figure(figsize=(14, 7))
    plt.plot(df[df['Date'] >= '2024-04-01']['Date'], df[df['Date'] >= '2024-04-01']['Close'], label='Actual Prices', color='black')
    plt.plot(future_df['Date'], future_df['Forecast'], label='Forecasted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Stock Price Forecast for {company_name} (April 1st to September 30th, 2024)')
    plt.legend()
    plt.show()