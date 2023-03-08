import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
import time
import itertools

load_data = np.loadtxt("Final/filterData.txt")
filter_data = load_data.reshape(load_data.shape[0], load_data.shape[1] // 48, 48)

class Battery:
    def __init__(self, maxSOC, maxChargeRate, maxDischargeRate, chargeEfficiency, dischargeEfficiency):
        self.maxSOC = maxSOC
        self.maxChargeRate = maxChargeRate
        self.maxDischargeRate = maxDischargeRate
        self.chargeEfficiency = chargeEfficiency
        self.dischargeEfficiency = dischargeEfficiency

peakLoad = max(loads)

SOC = max(loads)*0.25  # in kWh
D = max(loads)*0.25    # in kW
C = D/2                # in kW
efficiency = 0.95


batt = Battery(SOC, C, D, efficiency, efficiency)