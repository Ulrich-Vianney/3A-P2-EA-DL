#!/usr/bin/env python3

import argparse
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from numpy import exp, log
from torchdiffeq import odeint

all_planet_dict = {
    "Sun" : 0,
    "Mercury" : 1,
    "Venus" : 2,
    "Earth" : 3,
    "Mars": 4,
    "Jupiter": 5,
    "Saturn": 6,
    "Uranus" : 7,
    "Neptune" : 8,
    "Pluton" : 9
}

earth_only_dict = {
    "Sun" : 0,
    "Earth" : 1
}

telluric_only_dict = {
    "Sun" : 0,
    "Mercury" : 1,
    "Venus" : 2,
    "Earth" : 3,
    "Mars": 4
}

# Reference values for nondimensionalization
L_ref = 1.5e11  # Earth-Sun distance in meters
M_ref = 6e24  # Mass of the Earth in kg
T_ref = 3.15e7  # Orbital period of the Earth in seconds (1 year)

class OrbitalDynamics(nn.Module):
    def __init__(self, planet_dict = earth_only_dict,
                 ln_mass = None, initial_pos = None, initial_vel = None):
        super().__init__()
        self.planet_dict = planet_dict
        self.dim = len(self.planet_dict)
        self.ln_G = nn.Parameter(torch.tensor(-23.0))  # Gravitational constant
        
        # Initialize masses (nondimensionalized)
        if ln_mass is not None:
            self.ln_mass = ln_mass
        else:
            self.ln_mass = nn.Parameter(torch.tensor([12, 0.0]))  # Log masses (Sun and Earth)

        # Initialize positions (nondimensionalized)
        if initial_pos is not None:
            self.initial_pos = initial_pos
        else:
            self.initial_pos = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))  # Sun at origin, Earth at 1.0 in x-direction

        # Initialize velocities (nondimensionalized)
        if initial_vel is not None:
            self.initial_vel = initial_vel
        else:
            self.initial_vel = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0], [0.0, -6.28, 0.0]]))  # Earth moving in -y direction

    
    def forward(self, t, state):
        # state: tensor of shape (2 * self.dim, 3)
        # Split state into positions and velocities
        pos = state[:self.dim]
        vel = state[self.dim:]

        # Compute pairwise differences
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # Shape: (self.dim, self.dim, 3)
        dist = torch.norm(diff, dim=2)  # Shape: (self.dim, self.dim)
        dist_cubed = dist**3 + 1e-10  # Add small epsilon to avoid division by zero

        # Compute gravitational forces (nondimensionalized)
        G = torch.exp(self.ln_G - 3*log(L_ref) + 2*log(T_ref) + log(M_ref)) ## G (m^3.s^-2.kg^-1) nondimensionalized
        masses = torch.exp(self.ln_mass)
        forces = G * (masses.unsqueeze(0) * masses.unsqueeze(1)).unsqueeze(2) * diff / dist_cubed.unsqueeze(2)  # Shape: (self.dim, self.dim, 3)

        # Sum forces to get the total force on each planet
        total_force = torch.sum(forces, dim=1)  # Shape: (self.dim, 3)

        # Compute acceleration: a = F / m
        dvel = total_force / masses.unsqueeze(1)  # Shape: (self.dim, 3)
        dpos = vel  # Shape: (self.dim, 3)

        # Concatenate dpos and dvel to form the derivative of the state
        return torch.cat([dpos, dvel], dim=0)
    
    def simulate(self, times):
        state = torch.cat([self.initial_pos, self.initial_vel], dim=0)

        # Observation times (nondimensionalized)
        nondim_times = times/T_ref

        solution = odeint(self, state, nondim_times, atol=1e-8, rtol=1e-8) 
        #absolute error and relative error à régler
        trajectory = solution[:, :self.dim] # Extract positions from the solution

        return trajectory

# def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
#     global_step = min(global_step, decay_steps)
#     cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
#     decayed = (1 - alpha) * cosine_decay + alpha
#     return learning_rate * decayed


# def learning_rate_schedule(
#     global_step, warmup_steps, base_learning_rate, lr_scaling, train_steps
# ):
#     warmup_steps = int(round(warmup_steps))
#     scaled_lr = base_learning_rate * lr_scaling
#     if warmup_steps:
#         learning_rate = global_step / warmup_steps * scaled_lr
#     else:
#         learning_rate = scaled_lr

#     if global_step < warmup_steps:
#         learning_rate = learning_rate
#     else:
#         learning_rate = cosine_decay(
#             scaled_lr, global_step - warmup_steps, train_steps - warmup_steps
#         )
#     return learning_rate


# def set_learning_rate(optimizer, lr):
#     for group in optimizer.param_groups:
#         group["lr"] = lr


### Test


def test_simplified_model():
    # Define the model
    model = OrbitalDynamics()

    # Define time steps for simulation (in seconds)
    days = 365  # Simulate for 1 year
    seconds_per_day = 86400  # Number of seconds in a day
    times = torch.linspace(0, 3*days * seconds_per_day, 1000)  # 1000 time steps over 1 year

    # Simulate the system
    trajectory = model.simulate(times)

    # Plot the trajectories of both bodies
    plt.figure(figsize=(10, 5))

    # Sun trajectory
    plt.plot(trajectory[:, 0, 0].detach().numpy(), trajectory[:, 0, 1].detach().numpy(), '.', label="Sun", color="orange")

    # Earth trajectory
    plt.plot(trajectory[:, 1, 0].detach().numpy(), trajectory[:, 1, 1].detach().numpy(), label="Earth", color="blue")

    plt.xlabel("X Position (km)")
    plt.ylabel("Y Position (km)")
    plt.title("Trajectories of Sun and Earth")
    plt.legend()
    plt.grid()
    plt.show()

test_simplified_model()

# # Define the model and simulate
# model = OrbitalDynamics()
# times = torch.linspace(0, 100, 1000)  # Time steps
# trajectory = model.simulate(times)

# # Plot the trajectory of the first planet
# import matplotlib.pyplot as plt
# plt.plot(trajectory[:, 0, 0].detach().numpy(), trajectory[:, 0, 1].detach().numpy())
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Trajectory of the First Planet")
# plt.show()