# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 30-09-2025



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load dataset
file_path = 'Sunspots.csv'
data = pd.read_csv(file_path)

# Show column names to verify
print("Columns in dataset:", data.columns)
print(data.head())

# ---- Adjust column names here ----
# Try to guess: if 'Date' exists use it, else if 'Month' or 'Year' exists use that
if 'Date' in data.columns:
    date_col = 'Date'
elif 'Month' in data.columns:
    date_col = 'Month'
elif 'Year' in data.columns:
    date_col = 'Year'
else:
    raise KeyError("No suitable date column found! Please check dataset.")

# Try to guess value column
if 'Sunspots' in data.columns:
    value_col = 'Sunspots'
elif 'Monthly Mean Total Sunspot Number' in data.columns:
    value_col = 'Monthly Mean Total Sunspot Number'
elif 'SSN' in data.columns:
    value_col = 'SSN'
else:
    raise KeyError("No suitable sunspot value column found! Please check dataset.")

# Convert date column to datetime
data[date_col] = pd.to_datetime(data[date_col], infer_datetime_format=True, errors='coerce')
data.set_index(date_col, inplace=True)

# Use the selected sunspot values
sunspot_values = data[value_col]

# If dataset is daily, resample to monthly
monthly_sunspots = sunspot_values.resample('M').mean()

# ---- ADF Test ----
result = adfuller(monthly_sunspots.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

# ---- Train/Test Split ----
train_size = int(len(monthly_sunspots) * 0.8)
train, test = monthly_sunspots[:train_size], monthly_sunspots[train_size:]

# ---- ACF & PACF ----
fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

# ---- Fit AR Model ----
ar_model = AutoReg(train.dropna(), lags=13).fit()

# ---- Predictions ----
ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# ---- Plot Predictions vs Actual ----
plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data (Sunspots)')
plt.xlabel('Time')
plt.ylabel('Sunspots')
plt.legend()
plt.show()

# ---- Mean Squared Error ----
mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

# ---- Plot Train, Test, and Predictions ----
plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction (Sunspots)')
plt.xlabel('Time')
plt.ylabel('Sunspots')
plt.legend()
plt.show()
```

### OUTPUT:

#### GIVEN DATA:
```
Columns in dataset: Index(['Unnamed: 0', 'Date', 'Monthly Mean Total Sunspot Number'], dtype='object')
   Unnamed: 0        Date  Monthly Mean Total Sunspot Number
0           0  1749-01-31                               96.7
1           1  1749-02-28                              104.3
2           2  1749-03-31                              116.7
3           3  1749-04-30                               92.8
4           4  1749-05-31                              141.7
ADF Statistic: -10.497051662546147
p-value: 1.1085524921956021e-18
The data is stationary.
```
#### ACF - PACF:
<img width="683" height="526" alt="image" src="https://github.com/user-attachments/assets/3b7911b3-fef9-4427-8844-a1d689f97117" />




#### PREDICTION:
<img width="850" height="391" alt="image" src="https://github.com/user-attachments/assets/eccbe15e-c5b6-4753-b03c-8718d5934a59" />





Mean Squared Error (MSE): 4837.057868431518

#### FINIAL PREDICTION:
<img width="850" height="391" alt="image" src="https://github.com/user-attachments/assets/5fa2ab1d-4c7b-4403-9a98-9660021c5111" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
