import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from operator import sub, mul, pow
from scipy.signal import savgol_filter

def WidenPeakLate(F2, Pmax, i, W):
    if i+1+W > 47:
        return F2
    F2[i+1+W] = F2[i+1]
    diff = (F2[i] - F2[i+1])/(W+1)
    for j in range(W):
        F2[i+W-j] = F2[i+1+W] + (j+1)*diff
    return F2

def WidenPeakEarly(F2, Pmax, i, W):
    if i-1-W < 0:
        return F2
    F2[i-1-W] = F2[i-1]
    diff = (F2[i] - F2[i-1])/(W+1)
    for j in range(W):
        F2[i-W+j] = F2[i-1-W] + (j+1)*diff
    return F2

def FindBounds(data):
     UpperBound = []
     LowerBound = []
     for feeder in range(data.shape[0]):
          UpperBound.append(np.quantile(data[feeder], 0.75, axis=0))
          LowerBound.append(np.quantile(data[feeder], 0.25, axis=0))
     return np.asarray(UpperBound), np.asarray(LowerBound)

def Filter_F1(data):
    
    UpperBounds, LowerBounds = FindBounds(data)

    A_peak = 0.8
    h = 48

    forecast_F1 = np.empty((data.shape[0], data.shape[1], data.shape[2]))
    for feeder in range(data.shape[0]):
        for day in range(data.shape[1]):
            dayLoads = data[feeder][day]
            UpperBound = UpperBounds[feeder]
            LowerBound = LowerBounds[feeder]
            MidPoint = (UpperBound - LowerBound) / 2
            # print(dayLoads)
            # print(MidPoint)
            for i in range(h):
                if (dayLoads[i] - MidPoint[i]) > MidPoint[i]:
                    forecast_F1[feeder][day][i] = ((1-(UpperBound[i]-dayLoads[i])/(UpperBound[i]-MidPoint[i]))*A_peak)*dayLoads[i]
                else:
                    forecast_F1[feeder][day][i] = ((1-(MidPoint[i]-dayLoads[i])/(MidPoint[i]-LowerBound[i]))*A_peak)*dayLoads[i]
    
    return np.asarray(forecast_F1)

def Filter_F2(data):

    forecast_F2 = np.array(data)
    F1 = np.array(data)
    Wpeak = 2
    Npeak = 3

    for feeder in range(data.shape[0]):
        for day in range(data.shape[1]):
            for i in range(Npeak):
                Pmax = max(F1[feeder][day])
                Pmax_pos = np.argmax(F1[feeder][day])
                forecast_F2[feeder][day] = WidenPeakLate(forecast_F2[feeder][day], Pmax, Pmax_pos, Wpeak)
                forecast_F2[feeder][day] = WidenPeakEarly(forecast_F2[feeder][day], Pmax, Pmax_pos, Wpeak)
                F1[feeder][day][Pmax_pos] = 0
    return np.asarray(forecast_F2)

def Filter_F3(data):
    forecast_F3 = np.empty((data.shape[0], data.shape[1], data.shape[2]))

    for feeder in range(data.shape[0]):
        for day in range(data.shape[1]):
            forecast_F3[feeder][day] = savgol_filter(data[feeder][day], 11, 6)
    return np.asarray(forecast_F3)


load_data = np.loadtxt("Final/forecastData.txt")
forecast_data = load_data.reshape(load_data.shape[0], load_data.shape[1] // 48, 48)

print("Performing Filter F1")
forecast_F1 = Filter_F1(forecast_data)
print("Performing Filter F2")
forecast_F2 = Filter_F2(forecast_F1)
print("Performing Filter F3")
forecast_F3 = Filter_F3(forecast_F2)

data_reshaped = forecast_F3.reshape(forecast_F3.shape[0], -1)
np.savetxt("Final/filterData.txt", data_reshaped)

# y = list(range(48))
# plt.plot(y, forecast_F3[9][0], label="F3")
# plt.plot(y, forecast_data[9][0], label="orig")
# plt.legend()
# plt.show()