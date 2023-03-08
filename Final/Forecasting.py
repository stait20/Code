import pandas as pd
import numpy as np
import datetime
from sklearn.neural_network import MLPRegressor
import time

# Read csv file, convert time to datetime data type
df = pd.read_csv('Final/flex_networks.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], infer_datetime_format=True)

# Find day of week and hour of day
times = df['Timestamp']
DoW = []
HoD = []

for x in times:
    DoW.append(x.weekday())
    HoD.append(x.hour)

df['DoW'] = DoW
df['HoD'] = HoD

# Target columns are all feeders excepet for three that have too many zero value
targetColumns = list(df.columns[1:19]) + list(df.columns[21:22])
# Uses all weather and time predictor values
predictors = list(df.columns)[23:]
# Normalise predictor values
df[predictors] = df[predictors]/df[predictors].max()

# Split data into test and train
# Train from Jan - May
# Test on June
X = np.split(df[predictors].values, [7247,8687])
y = np.split(df[targetColumns].values, [7247,8687])

X_train = X[0]
X_test = X[1]
y_train = y[0]
y_test = y[1]

# Create NN model
print("Begin model fitting")
t = time.time()
mlp = MLPRegressor(hidden_layer_sizes=(9,9,9), activation='relu', solver='adam', max_iter=5000)
mlp.fit(X_train,y_train)
elapsed = time.time() - t
print ('Time to fit model:', elapsed)

# Output predicted values in June
data = mlp.predict(X_test)

# Reshape data into 3d array
data2 = []
for feeder in range(data.shape[1]):
    tempDays = []
    for day in range(0,data.shape[0],48):
        temp = []
        for i in range(48):
            temp.append(data[day+i][feeder])
        tempDays.append(temp)
    data2.append(tempDays)

data = np.asarray(data2)

# Reshape data to allow for saving as txt. Will be easy to reconstitute later
data_reshaped = data.reshape(data.shape[0], -1)
np.savetxt("Final/forecastData.txt", data_reshaped)