from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt

# Create mesh and define function space
mesh = UnitSquareMesh(10,10)
V=FunctionSpace(mesh, 'P',1)

# Define boundary condtion
u_D= Expression('1+x[0]*x[0]+2*x[1]*x[1]', degree= 2)

def boundary (x, on_boundary):
    return on_boundary

bc= DirichletBC(V, u_D, boundary)

# Define variation problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-10)
a = dot(grad(u), grad(v))*dx
L= f*v*dx

# Compute solution
u= Function(V)
solve( a == L, u, bc)

# Plot solution and mesh
plot(u)
plot(mesh)
interactive()

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Compute error in L@ norm
error_L2 = errornorm(u_D, u, 'L2')

#Compute maximum erro at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_max = ', error_L2)
print('error_max = ', error_max)

#Hold plot
plt.title("Poisson equation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
