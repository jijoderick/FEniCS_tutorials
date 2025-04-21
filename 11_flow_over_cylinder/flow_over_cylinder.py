from fenics import *
import numpy as np
from mshr import *

import matplotlib.pyplot as plt
# from dolfin import *

# Parameters
U_in = 1.0  # Inlet velocity
D = 0.1  # Diameter of the cylinder
rho = 1.0  # Density
Re = 100.0  # Reynolds number

mu = rho * U_in * D / Re  # Dynamic viscosity
nu = mu / rho  # Kinematic viscosity

# Create computational domain and mesh
Entry_length = 0.5  # Length of the domain
Exit_length = 2.0  # Length of the domain
height = 1.0
domain = Rectangle(Point(-Entry_length, -height/2.0), Point(Exit_length, height/2.0)) - Circle(Point(0, 0), D / 2)
mesh = generate_mesh(domain, 64)

# show the mesh (Optonall)
plot(mesh)
#plt.show()
plt.savefig('mesh.png', dpi=300)