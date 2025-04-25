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

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)  # Velocity space
Q = FunctionSpace(mesh, 'P', 1)  # Pressure space

W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))


'''
# Define boundary conditions
inlet = 'near(x[0], -Entry_length)'
outlet = 'near(x[0], Exit_length)'
walls = 'near(x[1], -height/2.0) || near(x[1], height/2.0)'
cylinder = 'on_boundary && (pow(x[0], 2) + pow(x[1], 2) < pow(D / 2, 2))' 
'''
# Define boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -Entry_length) and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Exit_length) and on_boundary
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], -height/2.0) or near(x[1], height/2.0)
class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return (pow(x[0], 2) + pow(x[1], 2) < pow(D / 2, 2)) and on_boundary
# Create boundary objects
inlet = Inlet()
outlet = Outlet()
walls = Walls()
cylinder = Cylinder()

# Define boundary conditions
inlet_profile = ('4.0*U_in*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')
bcu_inlet = DirichletBC(W.sub(0), Expression(inlet_profile, U_in=U_in, degree=2), inlet)
bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
bcp_outlet = DirichletBC(W.sub(1), Constant(0), outlet)

bcs = [bcu_inlet, bcu_walls, bcu_cylinder, bcp_outlet]
