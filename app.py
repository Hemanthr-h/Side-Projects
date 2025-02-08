from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', predicted_price=None)

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    ticker = symbol.upper() + ".NS"  # Assuming the stock is listed on NSE India

    # Calculate the previous day's date
    end_date = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')

    # Fetch historical data for the given symbol
    df = yf.download(ticker, start="2010-01-01", end=end_date)

    # Prepare the data
    df.reset_index(inplace=True)
    df['prev_shifted'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    # Prepare the feature matrix and target vector
    x = df[['Open', 'prev_shifted', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # Train the Linear Regression model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Predict the next day's Close price
    last_row = df.iloc[-1]
    next_day_features = np.array([[last_row['Open'], last_row['Close'], last_row['High'], last_row['Low'], last_row['Volume']]])
    next_day_prediction = regressor.predict(next_day_features)
    predicted_price = next_day_prediction[0]

    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
