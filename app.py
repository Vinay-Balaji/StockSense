import numpy as np 
import pandas as pd
import plotly.graph_objs as go  # For creating interactive plots
import plotly.express as px  # For quick and easy plots
import yfinance as yf  # For fetching stock data
import tensorflow as tf  # For building deep learning models
from tensorflow.keras.models import Sequential  # For creating a sequential model
from tensorflow.keras.layers import Dense, LSTM, Dropout  # For layers in the neural network
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # For scaling data
from sklearn.base import BaseEstimator, RegressorMixin  # For custom scikit-learn estimator
from sklearn.model_selection import RandomizedSearchCV  # For hyperparameter tuning
from sklearn.cluster import KMeans  # For clustering similar stocks
import streamlit as st  # For creating the web app
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # For sentiment analysis
from newsapi import NewsApiClient  # For fetching news articles
from datetime import timedelta  # For date manipulation

# Setting up the NewsAPI client with the provided API key
newsapi = NewsApiClient(api_key='f4be7526eb964942b90a89ebc18f4f75')

# Setting the date range for fetching stock data
start = '2010-01-01'
end = pd.Timestamp.today().strftime('%Y-%m-%d')

# Title for the Streamlit web application
st.title('StockSense')

# User input for the stock ticker symbol
user_input = st.text_input("Enter Stock Ticker", 'AAPL')

# Fetching historical stock data using yfinance
df = yf.download(user_input, start=start, end=end)

# Displaying a description of the stock data
st.subheader('Data from 2010 - Present')
st.write(df.describe())

# Plotting the closing price with 100-day and 200-day moving averages
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
st.write("This chart includes 100-day and 200-day moving averages. Moving averages are used to smooth out price data and identify trends. A crossover of these lines can signal potential buy or sell opportunities.")
df['MA100'] = df['Close'].rolling(100).mean()  # Calculating 100-day moving average
df['MA200'] = df['Close'].rolling(200).mean()  # Calculating 200-day moving average
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))  # Plotting closing price
fig.add_trace(go.Scatter(x=df.index, y=df['MA100'], mode='lines', name='100-Day MA'))  # Plotting 100-day MA
fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='200-Day MA'))  # Plotting 200-day MA
fig.update_layout(title='Closing Price with 100MA & 200MA', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)  # Displaying the plot

# Plotting the trading volume over time
st.subheader('Volume vs Time Chart')
st.write("This chart displays the trading volume over time, which indicates the number of shares traded. Higher volume often signifies strong interest and can correlate with price movements.")
fig = px.line(df, x=df.index, y='Volume', title='Volume vs Time')
st.plotly_chart(fig)  # Displaying the plot

# Plotting the distribution of closing prices
st.subheader('Distribution of Closing Prices')
st.write("This histogram shows the distribution of closing prices, providing insights into the most frequent price ranges and overall volatility.")
fig = px.histogram(df, x='Close', nbins=50, title='Distribution of Closing Prices', marginal="box", opacity=0.7)
st.plotly_chart(fig)  # Displaying the plot

# Plotting a heatmap of feature correlations
st.subheader('Heatmap of Feature Correlation')
st.write("This heatmap displays the correlation between different features in the dataset. A higher correlation indicates a stronger relationship, which can be useful for feature selection and understanding market behavior.")
corr = df.corr()  # Calculating correlation matrix
fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title='Feature Correlation Heatmap')
st.plotly_chart(fig)  # Displaying the plot

# Sentiment Analysis on recent news articles
st.subheader("News Sentiment Analysis")
st.write("This section analyzes the sentiment of recent news articles related to the stock.")

# Fetching recent news articles using NewsAPI
news_articles = newsapi.get_everything(q=user_input, language='en', sort_by='publishedAt', page_size=20)
analyzer = SentimentIntensityAnalyzer()  # Initializing sentiment analyzer
sentiment_scores = []  # List to store sentiment scores
articles_with_scores = 0  # Counter for articles with sentiment scores
max_articles = 5  # Maximum number of articles to analyze

for article in news_articles['articles']:
    title = article['title']  # Article title
    url = article['url']  # Article URL
    content = article['content']  # Article content
    description = article.get('description', '')  # Article description
    text_to_analyze = content if content else description if description else title  # Text to analyze for sentiment
    sentiment = analyzer.polarity_scores(text_to_analyze)  # Analyzing sentiment
    score = sentiment['compound']  # Compound sentiment score
    if score != 0.0:  # Only consider articles with non-zero sentiment score
        sentiment_scores.append(score)  # Append score to list
        st.write(f"[{title}]({url}) - Sentiment Score: {score}")  # Display article title and score
        articles_with_scores += 1
    if articles_with_scores >= max_articles:  # Stop if maximum articles analyzed
        break

# Calculating and displaying average sentiment score
if sentiment_scores:
    average_sentiment = np.mean(sentiment_scores)  # Calculate average sentiment
    st.write(f"Average Sentiment Score: {average_sentiment:.2f}")
else:
    st.write("No significant sentiment data available.")  # No sentiment data available
st.write("A positive score indicates positive sentiment, while a negative score indicates negative sentiment.")

# Incorporating sentiment into predictions
sentiment_adjustment = average_sentiment * 0.05  # Adjust prediction based on sentiment

# Splitting data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])  # First 70% of data for training
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])  # Last 30% for testing

# Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))  # Min-max scaling
data_training_array = scaler.fit_transform(data_training)  # Transform training data
scale_factor = 1 / scaler.scale_[0]  # Scale factor for inverse transformation

# Splitting data into x_train and y_train for LSTM model
x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])  # 100 previous days as features
    y_train.append(data_training_array[i,0])  # Current day as target
x_train, y_train = np.array(x_train), np.array(y_train)

# Custom Keras Regressor class for using with scikit-learn API
class CustomKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, units=50, dropout_rate=0.2, epochs=10, batch_size=32):
        self.units = units  # Number of LSTM units
        self.dropout_rate = dropout_rate  # Dropout rate for regularization
        self.epochs = epochs  # Number of training epochs
        self.batch_size = batch_size  # Batch size
        self.model = None  # Placeholder for the model
    
    def build_model(self):
        model = Sequential()  # Initializing the model
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # LSTM layer
        model.add(Dropout(self.dropout_rate))  # Dropout for regularization
        model.add(LSTM(units=self.units, return_sequences=False))  # Second LSTM layer
        model.add(Dropout(self.dropout_rate))  # Dropout for regularization
        model.add(Dense(units=1))  # Output layer for regression
        model.compile(optimizer='adam', loss='mean_squared_error')  # Compiling the model
        return model
    
    def fit(self, X, y):
        self.model = self.build_model()  # Building the model
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)  # Fitting the model
        return self
    
    def predict(self, X):
        return self.model.predict(X)  # Predicting with the model

# Using the custom regressor with hyperparameter tuning
model = CustomKerasRegressor()

# Defining the grid of hyperparameters to search
param_grid = {
    'units': [50],  # Number of LSTM units
    'dropout_rate': [0.2],  # Dropout rate
    'batch_size': [32],  # Batch size
    'epochs': [10]  # Number of epochs
}

# Setting up the grid search for hyperparameter tuning
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=3, n_jobs=-1, n_iter=3)

# Fitting the grid search to the data
grid_result = grid.fit(x_train, y_train)

# Retraining the model using the best parameters found
best_model = grid_result.best_estimator_
best_model.fit(x_train, y_train)

# Combining training and testing data for full prediction range
full_data = pd.concat([data_training, data_testing], ignore_index=True)
full_data_array = scaler.transform(full_data)

# Preparing x_full for prediction
x_full = []
for i in range(100, full_data_array.shape[0]):
    x_full.append(full_data_array[i-100: i])
x_full = np.array(x_full)

# Predicting the full data range using the best model
y_predicted_full = best_model.predict(x_full)
y_predicted_full = y_predicted_full * scale_factor  # Inverse scaling

# Adjusting predictions based on sentiment
y_predicted_full = y_predicted_full * (1 + sentiment_adjustment)

# Predicting future stock prices for the next 200 days
future_predictions = []
last_input = x_full[-1]

for _ in range(200):
    next_prediction = best_model.predict(np.expand_dims(last_input, axis=0))
    next_prediction = next_prediction * scale_factor  # Inverse scaling
    future_predictions.append(next_prediction.flatten()[0])
    last_input = np.roll(last_input, -1)  # Rolling input to include the next prediction
    last_input[-1] = next_prediction

# Creating future dates for the next 200 days
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 201)]

# Plotting predictions vs original data
st.subheader("Predictions vs Original")
st.write("This chart compares the original and predicted stock prices. The model's accuracy can be assessed by how closely the predicted prices follow the actual prices.")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Original Price', line=dict(color='blue')))
fig2.add_trace(go.Scatter(x=df.index[-len(y_predicted_full):], y=y_predicted_full.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
fig2.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predictions', line=dict(color='green')))
fig2.update_layout(title='Predictions vs Original', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig2)

# Stock Recommendation Feature: Finding similar stocks
def get_similar_stocks(ticker, num_clusters=5, recommendations=3):
    stock_symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'INTC']
    data = {symbol: yf.download(symbol, start=start, end=end) for symbol in stock_symbols}  # Fetching stock data

    features = []
    for symbol, df in data.items():
        if df is not None and not df.empty:
            avg_price = df['Close'].mean()  # Average closing price
            vol = df['Volume'].mean()  # Average volume
            features.append([avg_price, vol])

    scaler = StandardScaler()  # Scaling features
    features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=num_clusters)  # KMeans clustering
    clusters = kmeans.fit_predict(features)

    input_stock_data = yf.download(ticker, start=start, end=end)  # Fetching input stock data
    input_features = [[input_stock_data['Close'].mean(), input_stock_data['Volume'].mean()]]
    input_features = scaler.transform(input_features)  # Scaling input features
    input_cluster = kmeans.predict(input_features)[0]  # Predicting cluster

    # Filter stocks in the same cluster as the input stock, excluding the input stock itself
    similar_stocks = [stock_symbols[i] for i in range(len(clusters)) if clusters[i] == input_cluster and stock_symbols[i] != ticker]

    # If the number of similar stocks is less than the desired number of recommendations
    if len(similar_stocks) < recommendations:
        # Find additional stocks that are not the input stock and not already in the similar stocks list
        additional_stocks = [stock for stock in stock_symbols if stock != ticker and stock not in similar_stocks]
        # Add these additional stocks to the similar stocks list, up to the desired number of recommendations
        similar_stocks += additional_stocks[:recommendations - len(similar_stocks)]

    # Return the list of similar stocks, limited to the specified number of recommendations
    return similar_stocks[:recommendations]

# Integrating stock recommendations with Streamlit
similar_stocks = get_similar_stocks(user_input)
st.subheader("Similar Stocks Recommendations")
st.write(f"Stocks similar to {user_input}: {', '.join(similar_stocks)}")
