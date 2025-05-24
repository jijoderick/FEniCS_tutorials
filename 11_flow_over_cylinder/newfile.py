# I have readapted this from the fenics forum post by zhaoyang584520,(all credits to him)
from __future__ import print_function
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np

T = 2.0 # final time
dt = 0.00005

num_steps = int(T/dt) # number of time steps
mu = 0.001 # dynamic viscosity
rho = 1 # density
Re = 10 # Reynolds number

# Create mesh
channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 128)

#Define funnction spaces
V = VectorFunctionSpace(mesh, 'P',2)
Q = FunctionSpace(mesh, 'P', 1)
elem01 = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, elem01)

# Define boundaries
inflow = 'near(x[0], 0)'
outflow = 'near(x[0],2.2)'
walls = 'near(x[1],0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.15 && x[0]<0.25 && x[1]>0.15 && x[1]<0.25'

# Define inflow profile
inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

# Define boundary conditions
bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(W.sub(0), Constant((0,0)), walls)
bcu_cylinder = DirichletBC(W.sub(0), Constant((0,0)), cylinder)
bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)
bcs = [bcu_inflow, bcu_walls, bcu_cylinder, bcp_outflow]

dw = TrialFunction(W)
v, q = TestFunctions(W)
w = Function(W)
w_n = Function(W)
w_fem = Function(W)

du, dp = split(dw)
u, p = split(w)
u_n,p_n = split(w_n)
u_fem, p_fem = split(w_fem)

class InitialCondition(Expression):
    def eval(self, values, x):
        values[0] = 0.0  # u
        values[1] = 0.0  # v
        values[2] = 0.0  # p
    def value_shape(self):
        return (3,)

'''
class InitialCondition(Expression):
    def eval(self, values, x):

        if near(x[0], 0):
            values[0] = 4.0 * 1.5 * x[1] * (0.41 - x[1])/ pow(0.41, 2) # u
            values[1] = 0.0 # v
            values[2] = 0.0 # p
        else:
            values[0] = 0.0 # u
            values[1] = 0.0 # v
            values[2] = 0.0 # p
    def value_shape(self):
        return (3,)'''
    
initial_condition = InitialCondition(degree=2)
w_n.interpolate(initial_condition)
w.interpolate(initial_condition)

# Variational formulation
F1 = rho * dot((u - u_n)/dt, v)*dx \
     + rho * dot(dot(grad(u_n), u), v)*dx \
     + mu * inner(grad(u), grad(v))*dx \
     - p*div(v)*dx

F2 = div(u) * q * dx
F = F1 + F2
a = derivative(F, w, dw)

# Nonlinear problem and solver
class NS(NonlinearProblem):
    def __init__(self, a, L, bcs):
        NonlinearProblem.__init__(self)
        self.a = a
        self.L = L
        self.bcs = bcs

    def F(self, b, x):
        assemble(self.L, tensor=b)
        for bc in self.bcs:
            bc.apply(b)

    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A)

problem = NS(a, F, bcs)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6
solver.parameters["maximum_iterations"] = 60
solver.parameters["relaxation_parameter"] = 0.8

# Output
vtkfile_u = File("NS/velocity.pvd")
vtkfile_p = File("NS/pressure.pvd")

# Time-stepping
t = 0.0
for n in range(num_steps):
    t += dt
    w_n.vector()[:] = w.vector()
    solver.solve(problem, w.vector())
    u_, p_ = w.split()
    
    if n % 20 == 0:
        vtkfile_u << (u_, t)
        vtkfile_p << (p_, t)

        plot(u_, title="Velocity at t=%.2f" % t)
        plt.pause(0.01)
        plt.clf()