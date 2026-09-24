import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Define varriables 

R = 1000.0 # untis: ohms
C = 0.001 # units: farads
Qo = 1.0 # units: coulombs 
t_start = 0.0 # units: seconds
t_end = 10.0  # units: seconds
n_steps = 200     # number of steps
dt = (t_end - t_start) / n_steps # deleta t units: seconds 
t_eulers = np.linspace(0, t_end, n_steps)

# Analytical solution

def dQdt (t , Q , R , C):
     return -Q / (R * C)

def Qt_analytic (t , Qo , R , C):
     return Qo * np.exp(-t / (R * C))

# Euler Method 

    # f is the derivative function (dQdt) that gets passed as a variable 
    # (f is the value that represents the function since functions cannot be passed like numbers or strings)


def euler_method(f, y0, t_start, t_end, n_steps, *f_args):
    t = np.linspace(t_start, t_end, n_steps + 1)
    dt = t[1] - t[0] # finds that step size viia the gap between the first two time points
    Q = np.zeros(n_steps + 1)
    Q[0] = y0

    for i in range(n_steps):
        Q[i + 1] = Q[i] + dt * f(t[i], Q[i], *f_args)

    return t, Q

# Running the functions 

t_num, Q_num = euler_method(dQdt, Qo, t_start, t_end, n_steps, R, C) # running the function and defining the output
    # numerical (approximate) solution
Q_true = Qt_analytic(t_num, Qo, R, C) # runing the function and defining the output
    # actual solution
    # t_num is used in both so the graphs align 

# Plot Time Eulers vs Analytic 

plt.plot(t_num, Q_num, label='Euler')
plt.plot(t_num, Q_true, label='Analytic')
plt.ylabel("Charge (C)")
plt.xlabel("Time (s)")
plt.title("RC Discharge Curve")
plt.legend()
plt.show()
