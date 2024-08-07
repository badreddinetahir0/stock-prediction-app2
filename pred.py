import streamlit as st
import datetime
import yfinance as yf
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# Set start date and today's date
start = "2000-01-01"
today = datetime.date.today().strftime("%Y-%m-%d")

# Title of the app
st.title("Stock Prediction App")

# Dropdown for selecting stock
stocks = ('AAPL', 'GOOG', 'MSFT')
selected_stock = st.selectbox("Select dataset for predictions", stocks)

# Slider for selecting number of years for prediction
n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365

# Function to load data from Yahoo Finance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

# Load data
data_load_state = st.text("Load data ...")
data = load_data(selected_stock)
data_load_state.text("Loading data .. Done")

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecasted data
st.subheader('Forecasted data')
st.write(forecast.tail())

st.write('Forecasted data chart')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
