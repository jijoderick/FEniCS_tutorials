from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# tell the form compiler to use the C++ compiler optimization when compiling the code
parameters["form_compiler"]["cpp_optimize"]= True
# define a dictionary of options which can be passed to the compiler.
# The list of the optimization strategy will be used by the form compiler while generating the code
ffc_options = {"optimize": True}

# Create the mesh
mesh = UnitCubeMesh(24, 16, 16)
# Define the function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomains
left = CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

# Define the Dirichlet boundary (x= 0 or x=1)
c = Constant((0.0, 0.0, 0.0))
r = Expression(("scale*0.0",
                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3, degree = 2)

#The boundary subdomains and the boundary condition expressions areconstraints on the
# function space : math: V. The function space is therefore required as
# an argument to py class DirichletBC
bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)
bcs = [bcl, bcr]


# Defining the Trial function and trial function 
du = TrialFunction(V) # Incremental displacement
v  = TestFunction(V) # Test function 
u  = Function(V)     # displacement from the previous iteration
B  = Constant((0.0, 0.9, 0.0)) # Body force
T  = Constant((0.1, 0.0, 0.0)) # Traction force on the boundary

# Kinematic terms
d = len(u)
I = Identity(d) #Identity tensor
F = I + grad(u) # Deformation gradient
C = F.T* F      # Right Cauchy's tensor

# Invarients of Cauchy's tensor
Ic = tr(C)
J  = det(F)

# Elastic material parameter
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2+(1+nu))), Constant(E*nu/((1+nu)*(1-2*nu)))
# strain energy density (compressible Neo-Hookean model)
psi = (mu/2)*(Ic -3)- mu*ln(J) + (lmbda/2)*(ln(J)**2)

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T,u)*ds

# Compute the first derrivative of Pi
F =derivative(Pi, u, v)
# Compute the Jacobian of F
J = derivative(F, u, du)
# Solve the variational form
solve(F==0, u, bcs, J=J,
    form_compiler_parameters = ffc_options)

# Save the solution in VTK Format
file = File("displacement.pvd");
file << u; 


plot(u, mode = "displacement")
plt.show()  
