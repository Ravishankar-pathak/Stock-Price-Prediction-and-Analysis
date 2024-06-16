from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

app = Flask(__name__)


def fetch_stock_data(ticker, period='1mo'):
    if period.endswith('mo') or period.endswith('y'):
        data = yf.download(ticker, period=period)
    else:
        if period.endswith('d'):
            num_days = int(period[:-1])
            num_months = num_days // 30
            data = yf.download(ticker, period=f"{num_months}mo")
        else:
            data = yf.download(ticker, period=period)
    return data


def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data, scaler


def get_company_info(ticker):
    company = yf.Ticker(ticker)
    company_info = company.info
    company_name = company_info.get('longName', 'N/A')
    symbol = company_info.get('symbol', 'N/A')
    country = company_info.get('country', 'N/A')
    sector = company_info.get('sector', 'N/A')
    industry = company_info.get('industry', 'N/A')
    website = company_info.get('website', 'N/A')
    return company_name, symbol, country, sector, industry, website


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict_price(model, X_test):
    return model.predict(X_test)


def calculate_accuracy(actual_prices, predicted_prices):
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mean_actual_prices = np.mean(actual_prices)
    accuracy = (1 - (rmse / mean_actual_prices)) * 100
    return accuracy


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        period = request.form['period']
        stock_data = fetch_stock_data(ticker, period)
        historical_data = fetch_stock_data(ticker, '3mo')

        today_closing_price = stock_data['Close'].iloc[-1]
        today_closing_date = stock_data.index[-1].strftime('%Y-%m-%d')

        scaled_data, scaler = scale_data(historical_data['Close'])

        company_name, symbol, country, sector, industry, website = get_company_info(ticker)

        # Plot Closing Price vs. Time using Plotly
        closing_price_trace = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines',
                                         name='Closing Price')
        layout = go.Layout(title='Closing Price vs. Time',
                           xaxis=dict(title='Date'),
                           yaxis=dict(title='Closing Price'))
        closing_price_fig = go.Figure(data=[closing_price_trace], layout=layout)
        closing_price_fig.update_layout(hovermode='x unified')

        X_train = np.arange(len(scaled_data)).reshape(-1, 1)
        y_train = historical_data['Close'].values

        gb_model = GradientBoostingRegressor()
        svm_model = SVR()
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor()

        gb_model = train_model(gb_model, X_train, y_train)
        svm_model = train_model(svm_model, X_train, y_train)
        lr_model = train_model(lr_model, X_train, y_train)
        rf_model = train_model(rf_model, X_train, y_train)

        X_test = np.arange(len(historical_data)).reshape(-1, 1)

        gb_predicted_prices = predict_price(gb_model, X_test)
        svm_predicted_prices = predict_price(svm_model, X_test)
        lr_predicted_prices = predict_price(lr_model, X_test)
        rf_predicted_prices = predict_price(rf_model, X_test)

        actual_prices = historical_data['Close'].values

        gb_accuracy = calculate_accuracy(actual_prices, gb_predicted_prices)
        svm_accuracy = calculate_accuracy(actual_prices, svm_predicted_prices)
        lr_accuracy = calculate_accuracy(actual_prices, lr_predicted_prices)
        rf_accuracy = calculate_accuracy(actual_prices, rf_predicted_prices)

        # Plot actual vs. predicted prices
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=actual_prices, mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=gb_predicted_prices, mode='lines', name='GB Predicted Price'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=svm_predicted_prices, mode='lines', name='SVM Predicted Price'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=lr_predicted_prices, mode='lines', name='LR Predicted Price'))
        fig.add_trace(go.Scatter(x=stock_data.index, y=rf_predicted_prices, mode='lines', name='RF Predicted Price'))
        fig.update_layout(title='Actual vs. Predicted Prices', xaxis_title='Date', yaxis_title='Price')
        plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

        table_data = [{'date': date, 'actual_price': actual_price, 'gb_predicted_price': gb_predicted_price,
                       'svm_predicted_price': svm_predicted_price, 'lr_predicted_price': lr_predicted_price,
                       'rf_predicted_price': rf_predicted_price}
                      for
                      date, actual_price, gb_predicted_price, svm_predicted_price, lr_predicted_price, rf_predicted_price
                      in zip(historical_data.index[-30:], actual_prices, gb_predicted_prices, svm_predicted_prices,
                             lr_predicted_prices, rf_predicted_prices)]

        return render_template('integrated.html', today_closing_price=today_closing_price,
                               today_closing_date=today_closing_date, company_name=company_name, symbol=symbol,
                               country=country, sector=sector, industry=industry,
                               website=website,
                               closing_price_html=closing_price_fig.to_html(full_html=False, include_plotlyjs='cdn'),
                               table_data=table_data, period=period,
                               gb_accuracy=gb_accuracy, svm_accuracy=svm_accuracy, lr_accuracy=lr_accuracy,
                               rf_accuracy=rf_accuracy,
                               plot_div=plot_div)

    return render_template('integrated.html')


if __name__ == '__main__':
    app.run(debug=True)
