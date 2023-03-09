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

# Target columns are all feeders excepet for four that have too many zero value
targetColumns = list(df.columns[1:4]) + list(df.columns[5:19]) + list(df.columns[21:22])
# Create a 1 and 7 day persistance forecast for each feeder
for name in targetColumns:
    df[name + "_1day_pers"] = df[name].shift(48, axis=0)
    df[name + "_7day_pers"] = df[name].shift(336, axis=0)
# Uses all weather and time predictor values
predictorsAll = list(df.columns)[23:]
# Normalise predictor values
df[predictorsAll] = df[predictorsAll]/df[predictorsAll].max()

print("Begin model fitting")
t = time.time()
mlpAll = []
data = np.empty((18,1440))
i = 0
# Create seperate model for each feeder
for name in targetColumns:
    # Predictors are day of week, hour of day, air temp
    # and both persistance forecasts for that feeder
    predictors = ['DoW', 'HoD', name + "_1day_pers", name + "_7day_pers", 'Air Temperature']
    targetColumn = name

    # Split data into test and train
    # Train from Jan - May
    # Test on June
    X = np.split(df[predictors].values, [336,7247,8687])
    y = np.split(df[targetColumn].values, [336,7247,8687])

    X_train = X[1]
    X_test = X[2]
    y_train = y[1]
    y_test = y[2]

    print("Training model", i+1)
    mlp = MLPRegressor(hidden_layer_sizes=(5,5,5), activation='relu', solver='adam', max_iter=5000)
    mlp.fit(X_train,y_train)
    mlpAll.append(mlp)

    predict_test = mlp.predict(X_test)
    data[i] = predict_test
    i += 1

elapsed = time.time() - t
print ('Time to fit all models:', elapsed)

# Reshape data into 3d array
data2 = np.empty((18,30,48))
for feeder in range(data2.shape[0]):
    for day in range(data2.shape[1]):
        for hour in range(data2.shape[2]):
            data2[feeder][day][hour] = data[feeder][day*48 + hour]

data = np.asarray(data2)

# Reshape data to allow for saving as txt. Will be easy to reconstitute later
data_reshaped = data.reshape(data.shape[0], -1)
np.savetxt("Final/forecastData.txt", data_reshaped)