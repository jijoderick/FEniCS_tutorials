from fenics import *
import numpy as np
from mshr import *

# Parameters
U_in = 1.0  # Inlet velocity
D = 0.1  # Diameter of the cylinder
rho = 1.0  # Density
Re = 100.0  # Reynolds number

mu = rho * U_in * D / Re  # Dynamic viscosity
nu = mu / rho  # Kinematic viscosity

# Create computational domain and mesh