import argparse
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from numpy import exp, log
from torchdiffeq import odeint

all_planets = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

earth_only_list = ["Sun", "Earth"]

telluric_planet = ["Sun", "Mercury", "Venus", "Earth", "Mars"]

# Reference values for nondimensionalization
L_ref = 1.5e11  # Earth-Sun distance in meters
M_ref = 6e24  # Mass of the Earth in kg
T_ref = 3.15e7  # Orbital period of the Earth in seconds (1 year)


class OrbitalDynamics(nn.Module):
    def __init__(
        self,
        planet_list=earth_only_list,
        ln_mass=None,
        initial_pos=None,
        initial_vel=None,
        augmented_dim=0,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.planet_list = planet_list
        self.dim = len(self.planet_list)
        self.augmented_dim = augmented_dim
        self.device = device
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
            self.initial_pos = nn.Parameter(
                torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            )  # Sun at origin, Earth at 1.0 in x-direction

        # Initialize velocities (nondimensionalized)
        if initial_vel is not None:
            self.initial_vel = initial_vel
        else:
            self.initial_vel = nn.Parameter(
                torch.tensor([[0.0, 0.0, 0.0], [0.0, -6.28, 0.0]])
            )  # Earth moving in -y direction

        # Define a small auxiliary neural network for augmented dynamics
        if self.augmented_dim > 0:
            self.aux_net = nn.Sequential(
                nn.Linear(3 + augmented_dim, 32),  # Input: state + augmented part
                nn.Tanh(),
                nn.Linear(32, 3 + augmented_dim),
            )  # Output: dynamics for augmented state
        else:
            self.aux_net = None

    def forward(self, t, state):
        # state: tensor of shape (2 * self.dim , 3+ self.augmented_dim)
        # Split state into positions, velocities, and augmented dimensions
        pos = state[: self.dim, : -self.augmented_dim]
        vel = state[self.dim :, : -self.augmented_dim]
        # augmented = state[:, -self.augmented_dim :]
        # print(pos.shape, vel.shape, augmented.shape)

        # Compute pairwise differences
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # Shape: (self.dim, self.dim, 3)
        print(diff.shape)
        dist = torch.norm(diff, dim=2)  # Shape: (self.dim, self.dim)
        dist_cubed = dist**3 + 1e-10  # Add small epsilon to avoid division by zero
        print(dist_cubed.shape)
        # Compute gravitational forces (nondimensionalized)
        G = torch.exp(
            self.ln_G - 3 * log(L_ref) + 2 * log(T_ref) + log(M_ref)
        )  ## G (m^3.s^-2.kg^-1) nondimensionalized
        masses = torch.exp(self.ln_mass)
        mass_products = (masses.unsqueeze(0) * masses.unsqueeze(1)).unsqueeze(2)

        mass_products = mass_products.expand(-1, -1, 3)
        print(mass_products.shape)
        forces = G * mass_products * diff / dist_cubed.unsqueeze(2)
        # forces = (
        #     G * (masses.unsqueeze(0) * masses.unsqueeze(1)).unsqueeze(2) * diff / dist_cubed.unsqueeze(2)
        # )  # Shape: (self.dim, self.dim, 3)

        # Sum forces to get the total force on each planet
        total_force = torch.sum(forces, dim=1)  # Shape: (self.dim, 3)

        # Compute acceleration: a = F / m
        dvel = total_force / masses.unsqueeze(1)  # Shape: (self.dim, 3)
        dpos = vel  # Shape: (self.dim, 3)

        derivative = torch.cat([dpos, dvel], dim=0)
        # # Compute dynamics for augmented dimensions using the auxiliary network
        # Concatenate dpos, dvel, and daugmented to form the derivative of the state
        if self.augmented_dim == 1:
            # Concatenate positions and augmented state as input to the auxiliary network
            daugmented = self.aux_net(state)[:, -1].unsqueeze(1)  # Shape: (2*self.dim, augmented_dim)
            return torch.cat([derivative, daugmented], dim=1)
        elif self.augmented_dim > 1:
            daugmented = self.aux_net(state)[:, -self.augmented_dim :]
            return torch.cat([derivative, daugmented], dim=1)
        else:
            return derivative

    def simulate(self, times):
        # Initialize augmented dimensions to zero
        if self.augmented_dim == 0:
            state = torch.cat([self.initial_pos, self.initial_vel], dim=0)
        elif self.augmented_dim > 0:
            augmented_state = torch.zeros(2 * self.dim, self.augmented_dim)
            # .to(self.device)
            original_state = torch.cat([self.initial_pos, self.initial_vel], dim=0)
            state = torch.cat([original_state, augmented_state], dim=1)

        # Observation times (nondimensionalized)
        nondim_times = times / T_ref

        solution = odeint(self, state, nondim_times, atol=1e-8, rtol=1e-8)
        # absolute error and relative error à régler
        trajectory = solution[:, : self.dim, : -self.augmented_dim]  # Extract positions from the solution

        return trajectory


def test_simplified_model():
    # Define the model with augmented dimensions and auxiliary network
    model = OrbitalDynamics(augmented_dim=1)

    # Define time steps for simulation (in seconds)
    days = 365  # Simulate for 1 year
    seconds_per_day = 86400  # Number of seconds in a day
    times = torch.linspace(0, 3 * days * seconds_per_day, 1000)  # 1000 time steps over 1 year

    # Simulate the system
    trajectory = model.simulate(times)

    # Plot the trajectories of both bodies
    plt.figure(figsize=(10, 5))

    # Sun trajectory
    plt.plot(
        trajectory[:, 0, 0].detach().numpy(), trajectory[:, 0, 1].detach().numpy(), ".", label="Sun", color="orange"
    )

    # Earth trajectory
    plt.plot(trajectory[:, 1, 0].detach().numpy(), trajectory[:, 1, 1].detach().numpy(), label="Earth", color="blue")

    plt.xlabel("X Position (km)")
    plt.ylabel("Y Position (km)")
    plt.title("Trajectories of Sun and Earth")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_simplified_model()
