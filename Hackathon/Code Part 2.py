import numpy as np
import matplotlib.pyplot as plt

# Generate sample data for two graphs
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x) # Introduce a phase shift to make the graphs slightly different

# Calculate cross-correlation
cross_corr = np.correlate(y1, y2, mode='full')

# Find the lag for the maximum correlation
lag = np.argmax(cross_corr) - len(y1) + 1

# Calculate similarity score
similarity_score = cross_corr.max() / len(y1)

# Plot the original graphs
plt.figure(figsize=(10, 5))

plt.subplot(3, 1, 1)
plt.plot(x, y1, label='Graph 1')
plt.title('Graph 1')

plt.subplot(3, 1, 2)
plt.plot(x, y2, label='Graph 2')
plt.title('Graph 2')

# Plot the cross-correlation result
plt.subplot(3, 1, 3)
plt.plot(np.arange(-len(y1)+1, len(y1)), cross_corr, label='Cross-correlation')
plt.title('Cross-correlation between Graph 1 and Graph 2')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.axvline(x=lag, color='r', linestyle='--', label=f'Max Correlation Lag: {lag}')

plt.tight_layout()
plt.show()

# Create a new graph representing the similarity
plt.figure(figsize=(8, 4))
plt.plot(x, y1, label='Graph 1')
plt.plot(x, y2, label='Graph 2')
plt.title('Graphs with Similarity Highlights')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Highlight the aligned points based on the lag
plt.scatter(x, y1, c='blue', label='Graph 1')
plt.scatter(x - lag * (x[1] - x[0]), y2, c='orange', label='Graph 2 (Aligned)')

plt.legend()
plt.show()

# Output the similarity score
print(f"Similarity Score: {similarity_score}")

"""import numpy as np
import matplotlib.pyplot as plt

# Sample data for two curves
x_values = np.linspace(0, 10, 100)
y_values_curve1 = np.sin(x_values)
y_values_curve2 = np.cos(x_values)   # Shifted version for demonstration

# Plot the two curves
plt.plot(x_values, y_values_curve1, label='Curve 1')
plt.plot(x_values, y_values_curve2, label='Curve 2')

# Check for perfect overlap
overlap = np.allclose(y_values_curve1, y_values_curve2, rtol=0, atol=1e-10)

if overlap:
    print("The curves overlap perfectly.")
else:
    print("The curves do not overlap perfectly.")

plt.legend()
plt.show()"""

"""import yfinance as yf
import matplotlib.pyplot as plt

# Set the stock symbols and date ranges
symbol1 = 'SBUX'
symbol2 = 'HD'
start_date = '2021-01-01'
end_date = '2022-01-01'

# Download historical data
data1 = yf.download(symbol1, start=start_date, end=end_date)
data2 = yf.download(symbol2, start=start_date, end=end_date)

# Plot the two graphs side by side
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(data1['Close'], label=symbol1)
plt.title(f'{symbol1} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(data2['Close'], label=symbol2)
plt.title(f'{symbol2} Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

plt.tight_layout()
plt.show()

# Compare statistical measures
mean_diff = data1['Close'].mean() - data2['Close'].mean()
std_diff = data1['Close'].std() - data2['Close'].std()

print(f"Mean Difference: {mean_diff}")
print(f"Standard Deviation Difference: {std_diff}")"""

"""import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set the stock symbols and date range
symbol1 = 'AAPL'
symbol2 = 'MSFT'
start_date = '2021-01-01'
end_date = '2022-01-01'

# Download historical data
data1 = yf.download(symbol1, start=start_date, end=end_date)
data2 = yf.download(symbol2, start=start_date, end=end_date)

# Visual Inspection
plt.figure(figsize=(12, 6))
plt.plot(data1['Close'], label=symbol1)
plt.plot(data2['Close'], label=symbol2)
plt.title('Stock Price Comparison')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Correlation Analysis
correlation_coefficient = np.corrcoef(data1['Close'], data2['Close'])[0, 1]
print(f"Correlation Coefficient: {correlation_coefficient}")

# Regression Analysis
X = data1['Close'].values.reshape(-1, 1)
y = data2['Close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')"""

"""import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw

# Set the stock symbols and date range
symbol1 = 'AAPL'
symbol2 = 'MSFT'
start_date = '2021-01-01'
end_date = '2022-01-01'

# Download historical data
data1 = yf.download(symbol1, start=start_date, end=end_date)
data2 = yf.download(symbol2, start=start_date, end=end_date)

# Extract Close prices
prices1 = data1['Close'].values
prices2 = data2['Close'].values

# Perform dynamic time warping
distance, path = fastdtw(prices1, prices2)

# Visualize the alignment
plt.figure(figsize=(12, 6))
plt.plot(prices1, label=symbol1)
plt.plot(prices2, label=symbol2)
plt.title('Stock Price Comparison with DTW Alignment')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()

# Highlight the aligned points
for i, j in path:
    plt.plot([i, j], [prices1[i], prices2[j]], color='red', linestyle='--')

plt.show()

print(f"DTW Distance: {distance}")"""

"""import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Set the stock symbols and date range
symbol1 = 'AAPL'
symbol2 = 'MSFT'
start_date = '2021-01-01'
end_date = '2022-01-01'

# Download historical data
data1 = yf.download(symbol1, start=start_date, end=end_date)
data2 = yf.download(symbol2, start=start_date, end=end_date)

# Set the range of x-axis values (assuming indices represent time)
start_index = 50
end_index = 150

# Extract relevant data within the specified range
x_values = np.arange(start_index, end_index)
y_values1 = data1['Close'].values[start_index:end_index]
y_values2 = data2['Close'].values[start_index:end_index]

# Plot the selected range for both stocks
plt.figure(figsize=(12, 6))
plt.plot(x_values, y_values1, label=symbol1)
plt.plot(x_values, y_values2, label=symbol2)
plt.title('Stock Price Comparison')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Compare the relative changes between the two graphs
relative_changes = (y_values1 - y_values1[0]) / y_values1[0] - (y_values2 - y_values2[0]) / y_values2[0]

# Visualize the relative changes
plt.figure(figsize=(12, 4))
plt.plot(x_values, relative_changes, label='Relative Changes')
plt.axhline(y=0, color='red', linestyle='--', label='Zero Line')
plt.title('Relative Changes Comparison')
plt.xlabel('Date')
plt.ylabel('Relative Changes')
plt.legend()
plt.show()"""

"""# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with your actual file)
df = pd.read_csv('your_dataset.csv')

# Data preprocessing
# ... (handle missing data, feature engineering, etc.)

# Feature selection
# ... (choose relevant features)

# Define features (X) and target variable (y)
X = df[['Feature1', 'Feature2', 'Feature3']]  # Replace with actual feature names
y = df['TargetVariable']  # Replace with actual target variable name

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection and training (Random Forest Regressor as an example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Visualize predicted vs. actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

# Feature importance
feature_importance = model.feature_importances_
print('Feature Importance:')
for feature, importance in zip(X.columns, feature_importance):
    print(f'{feature}: {importance}')"""





