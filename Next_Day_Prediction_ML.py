import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import ta

ticker = 'TATAMOTORS.NS'
start_date = '1999-01-01'
end_date = '2024-04-22'
df = yf.download(ticker, start=start_date, end=end_date)
df.to_csv("Tatastockdfset.csv")

df = pd.read_csv("Tatastockdfset.csv", index_col=False)
missing_count = df.isna().sum()
print(missing_count)

# Exponential Moving Average (EMA)
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# Ichimoku Cloud components
nine_period_high = df['High'].rolling(window=9).max()
nine_period_low = df['Low'].rolling(window=9).min()
df['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2
twenty_six_period_high = df['High'].rolling(window=26).max()
twenty_six_period_low = df['Low'].rolling(window=26).min()
df['Kijun_sen'] = (twenty_six_period_high + twenty_six_period_low) / 2
df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
df['Senkou_span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
df['Chikou_span'] = df['Close'].shift(-26)

# Keltner Channels
keltner_channel = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20, window_atr=10)
df['Keltner_Channel_hband'] = keltner_channel.keltner_channel_hband()
df['Keltner_Channel_lband'] = keltner_channel.keltner_channel_lband()
df['Keltner_Channel_mband'] = keltner_channel.keltner_channel_mband()

# Pivot Points
P = (df['High'] + df['Low'] + df['Close']) / 3
df['Pivot_Point'] = P

df['Date'] = pd.to_datetime(df['Date'])
df['Weekday'] = df['Date'].dt.day_name()

df.to_csv("enhanced_features.csv")
df = pd.read_csv("enhanced_features.csv", index_col=False)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

plt.figure(figsize=(100, 50))
plt.plot(df['Date'], df['Open'], marker='o')
plt.title('Open Values by Weekday')
plt.xlabel('Date')
plt.ylabel('Open Value')
plt.grid(True)
plt.xticks(df['Date'], df['Weekday'], rotation=45)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.drop(['Date', 'Weekday',"Unnamed: 0"], axis=1).corr()


plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix with Coolwarm Theme')
plt.show()

X = df.drop(columns=["Unnamed: 0","Weekday","Date"])
y = df.drop(columns=['Open', 'Close', 'High', 'Low',"Unnamed: 0","Weekday","Date"])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# Initialize and fit models
models = {
    'GradientBoostingRegressor': MultiOutputRegressor(GradientBoostingRegressor(random_state=0)),
    'RandomForestRegressor': MultiOutputRegressor(RandomForestRegressor(random_state=0))
}
param_grids = {
    'GradientBoostingRegressor': {'estimator__n_estimators': [100, 200], 'estimator__learning_rate': [0.05, 0.1], 'estimator__max_depth': [3, 5]},
    'RandomForestRegressor': {'estimator__n_estimators': [10, 50], 'estimator__max_features': ['auto', 'sqrt']}
}

best_models = {}
results = {}
for name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[name], cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    results[name] = grid_search.best_estimator_.predict(X_test)

# Evaluate models
for name, predictions in results.items():
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{name}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}\n")