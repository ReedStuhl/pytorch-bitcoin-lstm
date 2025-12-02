# Bitcoin Price Prediction with PyTorch LSTM  
A modern LSTM time series forecasting project using PyTorch, real-time data, and clean ML engineering.

## Overview
This project predicts **Bitcoin's next closing price** using a PyTorch LSTM neural network.  
Originally built with TensorFlow/Keras years ago, this updated version uses:

- Modern **PyTorch** training loops  
- Clean **data preprocessing** and scaling  
- A **sliding window sequence dataset**  
- Real-time or CSV-based inputs  
- RMSE evaluation  
- Optional Binance/CoinGecko live data  

## Features

### Modern PyTorch LSTM Model
- 1-layer LSTM with 32 hidden units  
- Fully connected regression head  
- Predicts **next close price**  
- Clean training loop with Adam optimizer  

### Clean Dataset Pipeline
- Uses **Close** and **log returns** as features  
- Scales inputs and targets separately  
- Automatic sequence creation  
- Simple Dataset + DataLoader classes  

### Real-Time Bitcoin Price Mode  
The model can run with **live price data** instead of CSV using:

- Binance API (1m candles)
- CoinGecko API (free)

### Visualization & Evaluation
- Actual vs Predicted plots  
- Train and Test RMSE
