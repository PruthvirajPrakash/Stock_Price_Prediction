#Stock Price Prediction Project :

Project Overview :
This project aims to predict stock price of next day using historical data of Tata Motors (TATAMOTORS.NS) obtained from Yahoo Finance. 
The predictions from this project can be utilized in algorithmic trading, investment planning, and risk management.

Dataset Description :
This dataset consists of historical stock data for Tata Motors (TATAMOTORS.NS) sourced from Yahoo Finance.

1. It includes data ranging from January 1, 1999, to April 22, 2024, encompassing approximately 6300 trading days.
2. The dataset features columns for Date, Open, High, Low, Close, Adjusted Close, and Volume, providing a comprehensive view of stock activity.
3. Additional technical indicators have been calculated and added, including Exponential Moving Average (EMA), Ichimoku Cloud components, Keltner Channels, and Pivot Points.
4. This data set does not account for all market influences such as macroeconomic indicators or global financial events, which could affect the analysis.
5. If you wish to explore or utilize this dataset for your projects, you can access the raw and processed data files in the repository or download them from Yahoo Finance directly using the yfinance Python library.

Purpose of Selecting this Dataset:
1. The Tata Motors stock data from Yahoo Finance is instrumental for analyzing stock price movements and identifying market trends. This can aid investors in making informed decisions regarding buying or selling stocks.
2. The aim is to develop a predictive model that forecasts future stock prices based on historical data. This involves using technical indicators that are known to reflect underlying market sentiments and dynamics.
3. My personal interest in financial markets and algorithmic trading motivated the choice of this dataset. With these predictions, I hope to enhance trading strategies and improve investment returns.
4. Predictive modeling in stock trading can be particularly potent, providing insights that are not readily apparent through traditional investment analysis methods.
By integrating this model into a real-time trading system, it could serve to automate trading decisions, thus maximizing efficiency and potentially increasing profitability in volatile markets.


Features:
1. Numerical Features :
The dataset includes several numerical features that are pivotal for analyzing and predicting stock prices:
Price-related features: Open, High, Low, Close, and Adjusted Close prices provide a comprehensive view of the stock's daily trading range and final settlement price.
Volume: This indicates the total number of shares traded, offering insights into the liquidity and interest in the stock on a given day.
Technical Indicators:
EMA (Exponential Moving Average): Smooths out price data to identify trends.
Ichimoku Cloud Components: Includes various lines like Tenkan-sen, Kijun-sen, Senkou Span A & B, and Chikou Span, providing insights on potential support/resistance levels and momentum.
Keltner Channels: Highlight volatility and potential trend breakouts or reversals.

2. Categorical Features
While primarily focused on numerical data, the project also considers time-based categorical data:
Date: Used to trace the stock price movements over time and understand seasonal and temporal patterns.
Weekday: Extracted from the Date, this feature helps analyze if stock behaviors vary significantly across different days of the week.

Process Overview
The development of this project followed a systematic approach to ensure effective analysis and accurate stock price predictions for next day:

Step 1: Data Acquisition
Loaded historical stock data for Tata Motors (TATAMOTORS.NS) from Yahoo Finance covering a period from January 1, 1999, to April 22, 2024.

Step 2: Data Cleaning
Processed the data by checking for missing values and filled them using forward and backward filling methods.
Ensured data integrity by converting date strings into datetime objects for better manipulation and analysis.

Step 3: Exploratory Data Analysis (EDA)
Examined the dataset through various statistical summaries and visualizations to understand underlying patterns and behaviors.
Calculated additional technical indicators such as EMA, Ichimoku Cloud components, and Keltner Channels which are crucial for technical stock analysis.

Step 4: Feature Engineering
Derived new features like Pivot Points and added technical indicators to enrich the dataset.
Analyzed correlations among the features to identify significant predictors of stock prices.

Step 5: Data Transformation
Scaled the feature set using Standard Scaler to normalize data, ensuring that the model inputs have mean zero and variance one.
Visualized key relationships using scatter plots and correlation heatmaps to further refine the feature selection.

Step 6: Model Development and Validation
Split the data into training and testing sets to evaluate model performance.
Utilized machine learning techniques such as Gradient Boosting and Random Forest regressors.
Applied GridSearchCV with TimeSeriesSplit to optimize model parameters and validate the models rigorously to avoid overfitting.

Step 7: Model Evaluation
Assessed model performance using metrics like Mean Absolute Error (MAE) and R-squared.
Evaluated the model's ability to predict future stock prices and discussed potential improvements.

Exploratory Data Analysis (EDA)
Feature Selection
In this project, the dataset was enriched with technical indicators which serve as the features for our predictive models:

Open, High, Low, Close, Adjusted Close: These are fundamental stock market data points used daily by traders to analyze a stock's performance.
Volume: This represents the total number of shares traded during a given period and is a measure of the stock's liquidity.
Technical Indicators such as Exponential Moving Average (EMA), Ichimoku Cloud components (Tenkan-sen, Kijun-sen, Senkou Span A & B, Chikou Span), and Keltner Channels. These are used to predict future movements based on past price action and volume.
Target Variable
The target variable for this analysis is the Open,Close, High and Low price of the stock. 

Data Exploration
Statistical Summary: Conducted a statistical analysis to understand the central tendencies and dispersions of the stock prices and technical indicators.
Visualization: Plotted time series data to observe trends, cycles, and volatility in stock prices. Also, visualized relationships between features using scatter plots and histograms to identify patterns or anomalies in data distributions.
Correlation Analysis
Performed a correlation analysis to identify how closely changes in one feature are associated with changes in another. This helps in understanding which variables have the most influence on the target variable and can assist in feature selection for model building.
Handling Missing Values
Assessed and mitigated missing data within the dataset to ensure the robustness of the predictive models. Used forward fill and backward fill methods to maintain data continuity without introducing substantial bias.
By thoroughly analyzing these elements, the EDA process aids in making informed decisions on feature selection and further modeling steps, setting a strong foundation for the subsequent phases of model development.




   

