# StockSense: Stock Trend Prediction with Sentiment Analysis and Recommendations

## Overview

**StockSense** is a comprehensive tool designed to help investors analyze stock trends and make informed decisions. It leverages a combination of machine learning techniques and sentiment analysis to predict future stock prices and recommend similar stocks. The application integrates data from multiple sources, including historical stock data and news sentiment, to provide a holistic view of the stock market's behavior.

## Features

1. **Stock Trend Prediction**
   - Uses historical stock data to train a machine learning model (LSTM) for predicting future stock prices.
   - Incorporates moving averages (100-day and 200-day) to help identify trends and potential buy/sell signals.

2. **Sentiment Analysis**
   - Analyzes recent news articles related to a stock to gauge market sentiment.
   - Uses the VADER sentiment analysis tool to assign sentiment scores, which are then factored into the stock price prediction.

3. **Similar Stocks Recommendations**
   - Clusters stocks based on historical price and volume data.
   - Recommends stocks that exhibit similar characteristics to the user-input stock.

4. **Data Visualizations**
   - Provides charts for closing prices, moving averages, volume, price distribution, and feature correlation.
   - Visual representations help in understanding the stock's historical performance and market behavior.

## How It Works

1. **Data Collection and Processing**
   - Historical stock data is fetched using the yfinance library.
   - The data is processed and features such as closing prices, volume, and moving averages are computed.

2. **Model Training**
   - The LSTM model is trained on the processed data to predict future stock prices.
   - The model's performance is optimized using a grid search for hyperparameter tuning.

3. **Sentiment Adjustment**
   - Recent news articles related to the stock are analyzed for sentiment.
   - An average sentiment score is calculated and used to adjust the price predictions, accounting for recent market sentiment.

4. **Clustering for Recommendations**
   - Stocks are clustered using K-means based on their average price and volume.
   - Similar stocks are recommended based on the cluster of the input stock.

## Future Enhancements

1. **Enhanced Prediction Models:** Allowing users to set preferences and receive personalized recommendations.

2. **View Multiple Tickers At Once:** Allowing users to view analysis and predictions for multiple tickers at the same time.

3. **User Personalization:** Allowing users to set preferences and receive personalized recommendations.
