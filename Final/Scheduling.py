import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
import time
import itertools

def Run_Scheduler(batt, loadDict):
    # Initialise model
    m = ConcreteModel()

    # Create time to be used as index
    m.Time = RangeSet(0, 48-1)

    # Declare Decision variables
    m.SOC = Var(m.Time, bounds=(0,batt.maxSOC), initialize=0) # State of Charge variable, cant be greater than max SOC
    m.posDeltaSOC = Var(m.Time, initialize=0) # Change of State of Charge in kWh
    m.negDeltaSOC = Var(m.Time, initialize=0)
    m.chargingWatts = Var(m.Time, bounds=(0,batt.maxChargeRate), initialize=0) # Energy in grid, converted to Watts
    m.dischargingWatts = Var(m.Time, bounds=(0,batt.maxDischargeRate), initialize=0)
    m.gridImport = Var(m.Time, domain=NonNegativeReals, initialize=loadDict) # Net load from grid
    m.maxLoad = Var(within=NonNegativeReals)

    m.dayLoads = Param(m.Time, initialize=loadDict)
    m.chargingLimit = Param(initialize=batt.maxChargeRate)
    m.dischargingLimit = Param(initialize=batt.maxDischargeRate)

    # Constraints
    # SOC is equal to SOC at previous time plus change in SOC
    def SOC_rule(m,t):
        if t==0:
            return(m.SOC[t] == 0)
        else:
            return(m.SOC[t] == m.SOC[t-1] + m.posDeltaSOC[t] - m.negDeltaSOC[t])
    m.Batt_SOC = Constraint(m.Time, rule=SOC_rule)

    def charging_rule(m,t):
        return m.chargingWatts[t] <= m.chargingLimit
    m.charging_rule = Constraint(m.Time, rule=charging_rule)

    def discharging_rule(m,t):
        return m.dischargingWatts[t] <= m.dischargingLimit
    m.discharging_rule = Constraint(m.Time, rule=discharging_rule)

    # ensure charging rate obeyed
    def E_charging_rate_rule(m,t):
        return m.posDeltaSOC[t] == m.chargingWatts[t] * 0.5
    m.chargingLimit_cons = Constraint(m.Time, rule=E_charging_rate_rule)

    # ensure DIScharging rate obeyed
    def E_discharging_rate_rule(m,t):
        return m.negDeltaSOC[t] == m.dischargingWatts[t] * 0.5
    m.dischargingLimit_cons = Constraint(m.Time, rule=E_discharging_rate_rule)

    def demand_rule(m,t):
        return m.gridImport[t] == m.dayLoads[t] + m.chargingWatts[t] - m.dischargingWatts[t] 
    m.demand_rule = Constraint(m.Time, rule=demand_rule)

    def pos_or_neg_charge(m,t):
        return(m.chargingWatts[t] * m.dischargingWatts[t] == 0)
    m.pos_or_neg = Constraint(m.Time, rule=pos_or_neg_charge)


    def Peak_Rule(m, t):
        return m.maxLoad >= m.gridImport[t]

    m.Bound_Peak = Constraint(m.Time,rule=Peak_Rule)

    def Obj_func(m):
        return m.maxLoad
    m.max_load = Objective(rule=Obj_func, sense=minimize)

    opt = SolverFactory("gurobi")
    opt.options['NonConvex'] = 2

    t = time.time()
    opt.solve(m)
    elapsed = time.time() - t
    print ('Time elapsed:', elapsed)

    outputVars = np.zeros((9, 48))

    j = 0
    for v in m.component_objects(Var, active=True):
        varobject = getattr(m, str(v))
        for index in varobject:
            outputVars[j,index] = varobject[index].value
        j+=1
        if j>=9:
            break

    return outputVars

t = time.time()

load_data = np.loadtxt("Final/filterData.txt")
peakLoad = np.amax(load_data, axis=0)
filter_data = load_data.reshape(load_data.shape[0], load_data.shape[1] // 48, 48)

class Battery:
    def __init__(self, maxLoad, chargeEfficiency=0.95, dischargeEfficiency=0.95):
        self.maxSOC = maxLoad * 0.25                    # in kWh
        self.maxChargeRate = (maxLoad * 0.25) / 2       # in kW
        self.maxDischargeRate = maxLoad * 0.25          # in kW
        self.chargeEfficiency = chargeEfficiency
        self.dischargeEfficiency = dischargeEfficiency


batteries = []
for i in range(filter_data.shape[0]):
    batteries.append(Battery(peakLoad[i]))
    print(batteries[i].maxSOC)

models = []
for feeder in range(filter_data.shape[0]):
    for day in range(filter_data.shape[1]):
        loadDict = dict(enumerate(filter_data[feeder][day]))
        print("Solving model ", feeder*18 + day + 1, " out of 540")
        models.append(Run_Scheduler(batteries[feeder], loadDict))

models = np.asarray(models)
print(models.shape)
data_reshaped = models.reshape(models.shape[0], -1)

np.savetxt("Final/scheduleData.txt", data_reshaped)

elapsed = time.time() - t
print ('\nTotal time elapsed:', elapsed)