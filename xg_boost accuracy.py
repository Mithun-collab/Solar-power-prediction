import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import datetime
import time

df = pd.read_csv("niteditedfinal.csv")
df = df.fillna(0)
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])


train_df = df[df['Year'].isin([2017, 2018])]
test_df = df[df['Year'].isin([2019])]


X_train = train_df[['Year', 'Month', 'Day', 'Hour', 'Minute']].values
y_temp_train = train_df.iloc[:, 5].values
y_ghi_train = train_df.iloc[:, 6].values

X_test = test_df[['Year', 'Month', 'Day', 'Hour', 'Minute']].values
y_temp_test = test_df.iloc[:, 5].values
y_ghi_test = test_df.iloc[:, 6].values

model_temp = XGBRegressor()
model_temp.fit(X_train, y_temp_train)

model_ghi = XGBRegressor()
model_ghi.fit(X_train, y_ghi_train)

temp = model_temp.predict(X_test)
ghi = model_ghi.predict(X_test)

eta = 0.18
area = 7.4322
temp_factor = (1 - 0.05 * (temp - 25))
power = eta * area * ghi * temp_factor

test_df['temp_pred'] = temp
test_df['ghi_pred'] = ghi
test_df['power_pred'] = power

for i in range(100):
    print(f"Temperature: {temp[i]:.2f} °C")
    print(f"GHI: {ghi[i]:.2f} W/m²")
    print(f"Predicted Power: {power[i]:.2f} W")

mse_temp = mean_squared_error(y_temp_test, temp)
mae_temp = mean_absolute_error(y_temp_test, temp)
r2_temp = r2_score(y_temp_test, temp)

print("Mean squared error for temperature:", mse_temp)
print("Mean absolute error for temperature:", mae_temp)
print("R2 score for temperature:", r2_temp)

mse_ghi = mean_squared_error(y_ghi_test, ghi)
mae_ghi = mean_absolute_error(y_ghi_test, ghi)
r2_ghi = r2_score(y_ghi_test, ghi)

print("Mean squared error for GHI:", mse_ghi)
print("Mean absolute error for GHI:", mae_ghi)
print("R2 score for GHI:", r2_ghi)
