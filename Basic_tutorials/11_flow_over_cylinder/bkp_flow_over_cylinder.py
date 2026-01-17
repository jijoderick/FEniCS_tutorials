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

#W = MixedFunctionSpace([V, Q])  # Mixed space for velocity and pressure
#W = V * Q  # Mixed space for velocity and pressure
W = FunctionSpace(mesh, MixedElement([V.ufl_element(), Q.ufl_element()]))


# Define boundary conditions
def inflow(x, on_boundary):
    return on_boundary and near(x[0], -Entry_length)
def outflow(x, on_boundary):
    return on_boundary and near(x[0], Exit_length)
def walls(x, on_boundary):
    return on_boundary and (near(x[1], -height/2.0) or near(x[1], height/2.0))
def cylinder(x, on_boundary):
    return on_boundary and (x[0]**2 + x[1]**2 <= (D / 2)**2)

# Inlet velocity profile
inflow_velocity_profile = Expression(('U_in', '0'), degree=2, U_in=U_in)

# Define boundary conditions
bc_inflow = DirichletBC(W.sub(0), inflow_velocity_profile, inflow)
bc_outflow = DirichletBC(W.sub(1), Constant(0), outflow)
bc_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
bc_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
bcs = [bc_inflow, bc_outflow, bc_walls, bc_cylinder]
# Define trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
# Define functions for solution at previous and current time steps
u_n = Function(V)
p_n = Function(Q)
u_ = Function(V)
p_ = Function(Q)
# Define expressions used in variational forms
U = 0.5 * (u_n + u)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(0.01)  # Time step size
mu = Constant(mu)
rho = Constant(rho)

# Define strain- rate tensor
def epsilon(u):
    return sym(nabla_grad(u))
# Define stress tensor
def sigma(u, p):
    return 2 * mu * epsilon(u) - p * Identity(len(u))
# Define variational problem for step 1
F1 = rho * dot((u - u_n) / k, v) * dx + \
    rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx \
    + inner(sigma(U, p_n), epsilon(v)) * dx \
    + dot(p_n * n, v) * ds - dot(mu * nabla_grad(U) * n, v) * ds \
    - dot(f, v) * dx
a1 = lhs(F1)
L1 = rhs(F1)
# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q)) * dx
L2 = dot(nabla_grad(p_n), nabla_grad(q)) * dx - (1 / k) * div(u_) * q * dx
# Define variational problem for step 3
a3 = dot(u, q) * dx
L3 = dot(u_n, q) * dx - k * div(u_) * q * dx
# Create VTK file for saving solution
vtkfile = File('cylinder_flow.pvd')
# Time-stepping
t = 0
T = 2.0  # Total time
while t < T:
    # Step 1: Solve for velocity
    solve(a1 == L1, u_, bcs)
    # Step 2: Solve for pressure
    solve(a2 == L2, p_, bcs)
    # Step 3: Update velocity
    solve(a3 == L3, u_, bcs)
    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
    # Save solution to file
    vtkfile << (u_, t)
    t += float(k)
# Plotting the results
plt.figure()
plot(u_, title='Velocity')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('Velocity field')
plt.savefig('velocity.png', dpi=300)
plt.show()
plt.figure()
plot(p_, title='Pressure')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.title('Pressure field')
plt.savefig('pressure.png', dpi=300)
plt.show()
# Plotting the velocity field
