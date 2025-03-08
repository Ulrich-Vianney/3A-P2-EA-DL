from learn_orbital_physics import *
import pandas as pd
import numpy as np

def time_series(file):
    file = "3A-P2-EA-DL/csv/" + file  ## à adapter
    table = pd.read_csv(file)
    #print(table.columns)
    table["Date (TDB)"] = pd.to_datetime(table["Date (TDB)"])
    init_date = table["Date (TDB)"].values[0]
    table["Time"] = [(date-init_date).total_seconds() for date in table["Date (TDB)"]]
    #print(table.head(5))
    return table

def time_series_to_tensor(list_series):
    times = torch.Tensor(list_series[0]["Time"])
    pos_tensor = torch.Tensor([[[planet.iloc[time][pos] for pos in ["X (km)","Y (km)", "Z (km)"]] 
                                for planet in list_series] 
                                for time in range(len(list_series[0]))])
    vel_tensor = torch.Tensor([[[planet.iloc[time][vel] for vel in ["VX (km/s)","VY (km/s)", "VZ (km/s)"]] 
                                for planet in list_series] 
                                for time in range(len(list_series[0]))])
    print(times.shape, pos_tensor.shape, vel_tensor.shape)


earth = time_series("Earth.csv")
sun = time_series("Sun.csv")

nb_year = 2
nb_hours = int(nb_year*365.25*24)
hours_step = 24

observed_earth = earth.iloc[0:nb_hours:hours_step]
observed_sun = sun.iloc[0:nb_hours:hours_step]

time_series_to_tensor([sun, earth])