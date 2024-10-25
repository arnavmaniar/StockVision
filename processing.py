import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    df = df.dropna()
    return df

# model training init
def train_models(df: pd.DataFrame):
    X = df[['Year', 'Month', 'Day', 'DayOfWeek', 'IsMonthStart', 'IsMonthEnd', 'Lag1', 'Lag2', 'Lag3']]
    y = df['Close']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {'model': model, 'mse': mse, 'r2': r2, 'predictions': predictions, 'y_test': y_test}
        print(f'{name} - MSE: {mse}, R2: {r2}')
        
    return results

def forecast_prices(df: pd.DataFrame, model, start_date: str, end_date: str):
    future_dates = pd.date_range(start=start_date, end=end_date)
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['Day'] = future_df['Date'].dt.day
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['IsMonthStart'] = future_df['Date'].dt.is_month_start.astype(int)
    future_df['IsMonthEnd'] = future_df['Date'].dt.is_month_end.astype(int)
    
    last_known = df[df['Date'] <= start_date].iloc[-1]
    lag1 = last_known['Close']
    lag2 = df[df['Date'] <= start_date].iloc[-2]['Close']
    lag3 = df[df['Date'] <= start_date].iloc[-3]['Close']
    
    future_prices = []
    for i in range(len(future_df)):
        future_df.loc[i, 'Lag1'] = lag1
        future_df.loc[i, 'Lag2'] = lag2
        future_df.loc[i, 'Lag3'] = lag3
        forecast = model.predict(future_df[['Year', 'Month', 'Day', 'DayOfWeek', 'IsMonthStart', 'IsMonthEnd', 'Lag1', 'Lag2', 'Lag3']].iloc[i:i+1])[0]
        future_prices.append(forecast)
        lag3 = lag2
        lag2 = lag1
        lag1 = forecast
    
    future_df['Forecast'] = future_prices
    return future_df