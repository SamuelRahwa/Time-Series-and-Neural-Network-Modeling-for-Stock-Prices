# Time Series Modeling vs Neural Network Modeling for Stock Prices


**Author**: Samuel Rahwa


February 11, 2022


# Overview


I have been hired as Associate Quanitative Analyst for an upcoming Quant Hedge Fund.

I have been tasked with helping them to predict and forecast stock prices for the next 90 days.  

I need create models that reduce errors in their predicitions and make clear forecasts. 


# Business Problem

* Streamlining stock price forecasting to build a portfolio of potentially high growth rate stocks. Price will be forecasted over 90 days and percentage growth recorded for each.

* The models trained and functions developed will be tuned specifically for stock data. These functions will help produce results for an expanded selection of stocks.


# Data

* I used 6 years of historical adjusted closing stock prices for The Coca-Cola Company (KO) and American Airlines (AAL).

    - Only the Adjusted Closing Price for both Time Series Models and Neural Networks.
        * Volume will be the exogenous variable for the Time Series Models.
        * For the Sequential Data in Neural Networks, we are interesting in using the prices of the past 5 days to forecast 
          those of the future 1 day. This data structure is called many-to-many. 
          
    - The main advantage of adjusted closing prices is that they make it easier to evaluate stock performance. 
        * Firstly, the adjusted closing price helps investors understand how much they would have made by investing in a given
          asset. Most obviously, a 2-for-1 stock split does not cause investors to lose half their money. Since successful stocks
          often split repeatedly, graphs of their performance would be hard to interpret without adjusted closing prices.
          
    - Secondly, the adjusted closing price allows investors to compare the performance of two or more assets. 
        * Aside from the clear issues with stock splits, failing to account for dividends tends to understate the profitability of
          value stocks and dividend growth stocks. Using the adjusted closing price is also essential when comparing the returns
          of different asset classes over the long term. For example, the prices of high-yield bonds tend to fall in the long run.
          That does not mean these bonds are necessarily poor investments. Their high yields offset the losses and more, which can
          be seen by looking at the adjusted closing prices of high-yield bond funds.
          
    - However, there is a major criticism of using the Adjusted Closing Price. 
        * The nominal closing price of a stock or other asset can convey useful information. This information is destroyed by
          converting that price into an adjusted closing price. In actual practice, many speculators place buy and sell orders at
          certain prices, such as $100. As a result, a sort of tug of war can take place between bulls and bears at these key
          prices. If the bulls win, a breakout may occur and send the asset price soaring. Similarly, a win for the bears can lead
          to a breakdown and further losses. The adjusted close stock price obscures these events.
          
* Provided by the Yahoo Finance Api and the Alpha Vantage Api

    - [Yahoo Finance](https://finance.yahoo.com/?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAG_gCJT1OvT6N3rw86iRSVb4e6yxv-VgP4mhUaWbD89UPyDTNKsgWNPMWxR55DyHd6uU-On0iURaKibxGTewEAfYiybQGlOIMcVEIhADMTtUmPh4W68A3dYxHmYpAgX4OQSzmWop10PA9doLoW2caoBUzT2_kh2UBHrvvv2i3ij1)
    
    - [Alpha Vantage](https://www.alphavantage.co/)


# Strategies for Modeling



### Time Series Models:


* Fit ARIMA, Auto-ARIMA and Facebook Prophet models to the stock being analyzed. The model with the lowest RMSE on the test data was then fit on the entire stock price data set to forecast the next 90 days. Growth rates were recorded for each.

* To begin, several helper/sub functions were created to easily replicate the modeling process on any stock price data. These functions include:

    - Preprocessing functions to log transform and/or prepare the data for fitting each model.
    
    - Train Test Split function to divide the data specifically for time series. By default, the first 75% of the data is used to
      train.
      
    - Functions to find starting p, d, q values for the base ARIMA model order.
        - For finding the initial d value, the data was differenced and measured for significance using Dickey-Fuller test.
        - For finding the initial p value, analyzed the PACF graphs using the differenced data.
        - For finding the initial q value, analyzed the ACF graphs using the differenced data as well.
        
    - ARIMA, Auto-ARIMA and Facebook Prophet modeling functions with functionality to plot, summarize, return RMSE or growth rate


### Neural Networks:


* Recursive Neural Network (RNN), Long Short-Term Model (LSTM) and Gated Recurrent Unit (GRU) are part of an important branch of deep learning that deliver superior predictions for sequential data. We will these models on the stocks being analyzed. I was able to record models with the lowest RMSE on the test data.  

* To begin, several functions were created to help with the modeling process on any stock price data. These functions include:

    - Preprocessing functions to 
        - Train Test Split function to divide the data specifically for Neural Networks.
            - By choice, all the daily prices from 2015 to 2021 were used to train.
            - By choice, all the daily prices for all of 2021 were used to test.
        - Normalizing the input data, this is necessary in RNN/LSTM/GRU.
        - Prepare the data for fitting each model.
      
    - Checking the accuracy of our models, by comparing RMSE scores for the following:
        - Models without Regularization
            - Running the model with the potential of over fitting and NO regularzation techniques
        - Models with Regularization
            - Using the dropout technique to control overfitting. The dropout technique randomly drops or deactivates some neurons
              for a layer during each iteration. 
        
    - RNN, LSTM, LSTM with Regularization, GRU and GRU with Regularization modeling functions with functionality to plot the
      predictions and return RMSE
    




#  Predictions and Conclusion



### Time Series Models:



* Fit ARIMA (KO)
    - Untransformed Model:
        - AIC of 2292.42
        - RMSE of 2.15
    - Log Transformed Model:
        - AIC of -8955.07
        - RMSE of 2.08
    - Forecast:
        - Expects a growth of 3.66 % after 90 days
 
* Auto-ARIMA (KO) 
    - Untransformed Model:
        - AIC = 2276.82 
        - RMSE = 3.20
    - Log Transformed Model:
        - AIC of -9001.34 
        - RMSE of 3.21
    - Forecast:
        - Expects a growth of 0.0 %. The graph shows a neutral growth rate over time.
        
* Facebook Prophet (KO)
    - Untransformed Model:
        - RMSE of 9.45
    - Log Transformed Model:
        - RMSE of 8.94
    - Forecast:
        - Expects a growth of -2.29 % after 90 days. It maintains a cyclical growth
          rate over time.



* Fit ARIMA (AAL)
    - Untransformed Model:
        - AIC of 3876.43
        - RMSE of 7.09
    - Log Transformed Model:
        - AIC of -5853.49
        - RMSE of 5.54
    - Forecast:
        - Expects a growth of 5.8 % after 90 days

* Auto-ARIMA (AAL) 
    - Untransformed Model:
        - AIC of 3890.05
        - RMSE of 4.01
    - Log Transformed Model:
        - AIC of -5976.43
        - RMSE of 3.93
    - Forecast:
        - Expects a growth of 0.32 %. The graph shows a neutral growth rate over
          time.
        
* Facebook Prophet (AAL)
    - Untransformed Model:
        - RMSE of 21.30
    - Log Transformed Model:
        - RMSE of 13.91
    - Forecast:
        - Expects a growth of 41.96 % after 90 days. 
        

        


### Neural Networks Models:



* Simple RNN (KO)
    - Prediction RMSE: 0.73 
        
* LSTM (KO) 
    - Prediction RMSE: 0.71
        
* LSTM with Regularization (KO)
    - Prediction RMSE: 0.77
        
* GRU (KO) 
    - Prediction RMSE: 0.73
        
* GRU withRegularization (KO)
    - Prediction RMSE: 1.51



* Simple RNN (AAL)
    - Prediction RMSE: 0.66

* LSTM (AAL) 
    - Prediction RMSE: 0.62 
 
* LSTM with Regularization (AAL)
    - Prediction RMSE: 0.63 

* GRU (AAL) 
    - Prediction RMSE: 0.68

* GRU with Regularization (AAL)
    - Prediction RMSE: 2.81 





# Next Steps for Improvements: 


**How could I improve stock price predictiona and forecasting with more time and resources?**


> I could use the window method to complete the AutoRegressive Forecasting Method for the RNN's, LSTM's and GRU's for comparison 

> Instead of Stock Price Prediction, I would transtion this project to predict stock price movement     

> Provide additonally factors that could influence stock price or stock price movement:

>> Sentiment analysis (Consumer Sentiment, Reddit's API or Twitter's API)
>> Economic Indicators (GDP, CPI and/or Treasury Yields)
>> Fundamental Data (Balance Sheet, Cash Flows and/or Income Statement)


> Considerations and Thoughts:

>> Stock price/movement prediction is an extremely difficult task. Personally I don't think any of the stock prediction models out there shouldn't be taken for granted and blindly rely on them. However models might be able to predict stock price movement correctly most of the time, but not always.

>> ARIMA are designed for time series data while RNN-based models are designed for sequence data. Because of this distinction, it’s harder to build RNN-based models out-of-the-box.

>> ARIMA models are highly parameterized and due to this, they don’t generalize well. Using a parameterized ARIMA on a new dataset may not return accurate results. RNN-based models are non-parametric and are more generalizable.

>> Depending on window size, data, and desired prediction time, LSTM models can be very computationally expensive.


***

# For More Information

Please review my full technical analysis in [Jupyter Notebook](https://github.com/SamuelRahwa/Time-Series-and-Neural-Network-Modeling-for-Stock-Prices/tree/main/Modeling) or my nontechnical analysis in [Presentation](https://github.com/SamuelRahwa/Time-Series-and-Neural-Network-Modeling-for-Stock-Prices/blob/main/Time%20Series%20Modeling%20vs%20Neural%20Network%20Modeling%20for%20Stock%20Prices.pdf).

For any additional questions, please contact **Samuel Rahwa at samuelaaronrahwa@gmail.com**


# Repository Structure

```
├── Data                                           <- Both sourced externally and generated from code
├── Images                                         <- Both sourced externally and generated from code
├── Modeling                                       <- Narrative documentations of my analysis in Jupyter notebooks
├── README.md                                      <- The top-level README for reviewers of this project
└── Presentation                                   <- PDF version of project presentation

```