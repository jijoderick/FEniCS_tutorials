from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

T=2.0 # final time
num_steps = 10 # number of time steps
dt = T/ num_steps # time step size
alpha = 3 # parameter alpha
beta = 1.2 # parameter beta

# Create mesh and define funtion space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V= FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D =Expression ('1+x[0]*x[0]+alpha*x[1]*x[1]+ beta*t',
                degree=2, alpha=alpha, beta = beta, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)

#Define variation problem
u = TrialFunction(V)
v = TestFunction(V)
f= Constant(beta -2 -2*alpha)

F= u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx

a, L =lhs(F), rhs(F)

# Time0-stepping
u = Function(V)
t = 0
for n in range(num_steps):
    # Update current time
    t += dt
    u_D.t = T
    
    # Compute solution 
    solve (a == L, u, bc)

    # Plot solution 
    plot(u)
    plt.title('time %.2f' %t)
    plt.show()

    # Compute error at veritces
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().array() - u.vector().array()).max()
    print ('t = %.2f: error = %.3g' %(t, error))

    # Update previous solution
    u_n.assign(u)

# Hold plot 
interactive()
