from numpy import *
from matplotlib import pyplot
import matplotlib.animation as manimation
import os, sys

# HW5 Skeleton

# Function for calculating Euclidean distance between birds
# Passing in two bird locations, each bird is vector size (1,2)
# y is first bird, y1 is second bird
def distance(y, y1):
    # dist = (y[:,0] - y[k,0]**2 + (y[:,1] - y[k,1])**2) where y is the (Nx2) vector of
    distance = sqrt((y[0]-y1[0])**2 + (y[1]-y1[1])**2)
    return distance

# Calculates the flock diameter
def diameter(y):
    distances = zeros((N,N))
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            distances[i,j] = distance(y[i],y[j])

    diameter = max(distances.reshape(-1,))
    return diameter

def RK4(f, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
    '''
    Carry out a step of RK4 from time t using dt

    Input
    -----
    f:  Right hand side value function (this is the RHS function)
    y:  state vector
    t:  current time
    dt: time step size

    food_flag:  0 or 1, depending on the food location
    alpha:      Parameter from homework PDF
    gamma_1:    Parameter from homework PDF
    gamma_2:    Parameter from homework PDF
    kappa:      Parameter from homework PDF
    rho:        Parameter from homework PDF
    delta:      Parameter from homework PDF


    Output
    ------
    Return updated vector y, according to RK4 formula
    '''

    # Task: Fill in the RK4 formula
    k1 = f(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k2 = f((y+(k1*(dt/2.0))), (t+(dt/2.0)), food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k3 = f((y+(k2*(dt/2.0))), (t+(dt/2.0)), food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
    k4 = f((y+(dt*k3)), (t+dt), food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)

    y = y + (((dt/6.0))*(k1+2.0*k2+2.0*k3+k4))

    return y


def RHS(y, t, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta):
    '''
    Define the right hand side of the ODE

    '''
    N = y.shape[0]
    f = zeros_like(y)

    # y is B(t)
    # food_flag:  0 if stationary, 1 if moving
    # alpha -> food location on circle?
    # gamma_1 -> food
    # gamma_2 -> follow
    # kappa -> safety of flock
    # rho -> repel strength
    # delta -> closeness before collision?

    # Task:  Fill this in by assigning values to f

    c = zeros((1,2))
    if food_flag == 0:
        c[0,0] = 0.0
        c[0,1] = 0.0
    else:
        c[0,0] = sin(alpha*t)
        c[0,1] = cos(alpha*t)

    # Working
    f_food = zeros_like(y)
    f_food[0] = gamma_1 * (c - y[0])

    # Working
    f_follow = zeros_like(y)
    f_follow = gamma_2 * (y[0] - y)


    f_flock = zeros_like(y)
    y_bar = sum(y / N, axis=0)
    f_flock = kappa * (y_bar - y)
    f_flock[0,0] = 0
    f_flock[0,1] = 0

    f_repel = zeros_like(y)
    neighbors = zeros((N,5))
    # Loop through flock and assemble list of birds by distance
    flock_distance = zeros((N,N))
    for i in range(N):
        #calculate distance for all birds around given bird and sort
        #take top five and put in neighbors
        for j in range(N):
            if (i != j):
                flock_distance[i,j] = distance(y[i], y[j])
            else:
                flock_distance[i,j] = Inf

    # Sort the flock_distance array columns
    sorted_distances_indexes = argsort(flock_distance, axis=1)
    neighbors = sorted_distances_indexes[:,0:5]

    # Extract first five column from sorted_distance
    # Sorted distance index gives first five indexes of closest birds
    # We want N[sorted_distance_index[0:5]] to get actual birds locations
    # Then put those birds into neighbors
    # Sum up the repelling equation for each bird using the neighbors

    #iterate over neighbors and generate sum of repel?
    # Outer loop is each bird
    for k in range(1,neighbors.shape[0]):
        # Inner loop is each five closest neighbors
        for i in range(neighbors.shape[1]):
            f_repel[k] += rho*( (y[k]-y[neighbors[k,i]]) / ((y[k] - y[neighbors[k,i]])**2 + delta) )


    f = f_food + f_follow + f_flock + f_repel
    return f


##
# Set up problem domain
t0 = 0.0        # start time
T = 10.0        # end time
nsteps = 50     # number of time steps

# Task:  Experiment with N, number of birds
N = 30

# Task:  Experiment with the problem parameters, and understand how each parameter affects the system
dt = (T - t0) / (nsteps-1.0)
# Original value 2.0
gamma_1 = 2.0
#Original value 8.0
gamma_2 = 8.0
#Original value 0.4
alpha = 0.4
# Original value 4.0
kappa = 4.0
# Original value 2.0
rho = 2.0
# Original value 0.5
delta = 0.25
food_flag = 1   # food_flag == 0: C(x,y) = (0.0, 0.0)
                # food_flag == 1: C(x,y) = (sin(alpha*t), cos(alpha*t))

# Intialize problem
y = random.rand(N,2)  # This is the state vector of each Bird's position.  The k-th bird's position is (y[k,0], y[k,1])
flock_diam = zeros((nsteps,))

# Initialize the Movie Writer
# --> The movie writing code has been done for you
FFMpegWriter = manimation.writers['ffmpeg']
writer = FFMpegWriter(fps=6)
fig = pyplot.figure(0)
pp, = pyplot.plot([],[], 'k+')
rr, = pyplot.plot([],[], 'r+')
# Adding leader bird coloring
ll, = pyplot.plot([],[], 'b+')
pyplot.xlabel(r'$X$', fontsize='large')
pyplot.ylabel(r'$Y$', fontsize='large')
pyplot.xlim(-3,3)       # you may need to adjust this, if your birds fly outside of this box!
pyplot.ylim(-3,3)       # you may need to adjust this, if your birds fly outside of this box!


# Begin writing movie frames
with writer.saving(fig, "movie_" + str(N) + "_" + str(gamma_1) + "_" + str(gamma_2) + "_" + str(alpha) + "_" + str(kappa) + "_" + str(rho) + "_" + str(delta) + "_" + ".mp4", dpi=1000):

    c = zeros((1,2))

    # First frame
    pp.set_data(y[1:,0], y[1:,1])
    rr.set_data(c[0,0], c[0,1])
    # Adding leader bird
    ll.set_data(y[0,0], y[0,1])
    writer.grab_frame()

    t = t0

    # Declaring food so it can be plotted
    if food_flag == 0:
        c[0,0] = 0.0
        c[0,1] = 0.0
    else:
        c[0,0] = sin(alpha*t)
        c[0,1] = cos(alpha*t)

    for step in range(nsteps):

        # Task: Fill in the code for the next two lines
        y = RK4(RHS, y, t, dt, food_flag, alpha, gamma_1, gamma_2, kappa, rho, delta)
        flock_diam[step] = diameter(y)
        t += dt

        # Food declaration
        c = zeros((1,2))
        if food_flag == 0:
            c[0,0] = 0.0
            c[0,1] = 0.0
        else:
            c[0,0] = sin(alpha*t)
            c[0,1] = cos(alpha*t)

        # Movie frame
        pp.set_data(y[1:,0], y[1:,1])
        rr.set_data(c[0,0], c[0,1])
        # Adding leader bird
        ll.set_data(y[0,0], y[0,1])
        writer.grab_frame()


# Task: Plot flock diameter
fig1 = pyplot.figure(1)
pyplot.plot(linspace(t0,t,nsteps), flock_diam)
pyplot.title("Time vs. Flock Diameter")
pyplot.xlabel("Time sec")
pyplot.ylabel("Flock diameter")
pyplot.savefig("diameter_plot_" + str(N) + "_" + str(gamma_1) + "_" + str(gamma_2) + "_" + str(alpha) + "_" + str(kappa) + "_" + str(rho) + "_" + str(delta) + "_" + ".png", dpi=1000)
