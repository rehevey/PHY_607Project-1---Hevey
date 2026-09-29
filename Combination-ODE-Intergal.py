import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


# Define varriables 

R = 1000.0 # untis: ohms
C = 0.001 # units: farads
Qo = 1.0 # units: coulombs 
t_start = 0.0 # units: seconds
t_end = 100.0  # units: seconds
n_steps = 5000     # number of steps
dt = (t_end - t_start) / n_steps # deleta t units: seconds 
t_eulers = np.linspace(0, t_end, n_steps)

k = 8.9875e9 # Coulomb's constant 
q1 = 1.0e-6  # charge 1 in Coulombs 
q2 = 1.0e-6 #charge 2 in Coulombs 
Ri = 0.001 # lower bound (starting seperation), meters 
r_max = 20.0 # truncated stand-in for infinity, meters 
n_points = 10000 


# Analytical solution

def dQdt (t , Q , R , C):
     return -Q / (R * C)

def Qt_analytic (t , Qo , R , C):
     return Qo * np.exp(-t / (R * C))


def integrand (r, q1, q2, k):
    return (k * q1 * q2) / (r**2)

def W_analytic(R, q1, q2, k):
    return (k * q1 * q2) / Ri 


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
plt.title("RC Discharge Curve - Euler")
plt.legend()
plt.show()

# Runge Kutta 4th Order Method 

def RK_method(f, y0, t_start, t_end, n_steps, *f_args):
    t = np.linspace(t_start, t_end, n_steps + 1)
    Q = np.zeros(n_steps + 1)
    Q[0] = y0

    for i in range(n_steps):
        h = t[i + 1] - t[i]
        F1 = h * f(t[i], Q[i], *f_args)
        F2 = h * f(t[i] + h/2, Q[i] + F1/2, *f_args)
        F3 = h * f(t[i] + h/2, Q[i] + F2/2, *f_args)
        F4 = h * f(t[i] + h, Q[i] + F3, *f_args)
        Q[i + 1] = Q[i] + (1/6) * (F1 + 2*F2 + 2*F3 + F4)

    return t, Q

t_rk, Q_rk = RK_method(dQdt, Qo, t_start, t_end, n_steps, R, C)

plt.plot(t_rk, Q_rk, label='RK4')
plt.plot(t_rk, Q_true, label='Analytic')
plt.ylabel("Charge (C)")
plt.xlabel("Time (s)")
plt.title("RC Discharge Curve - RK")
plt.legend()
plt.show()

plt.plot(t_num, Q_num, label='Euler')
plt.plot(t_rk, Q_rk, label='RK4')
plt.plot(t_rk, Q_true, label='Analytic')
plt.ylabel("Charge (C)")
plt.xlabel("Time (s)")
plt.title("RC Discharge Curve - Both")
plt.legend()
plt.show()

# Using Scipy 

def dQdt_sys(t, y):
    return [-y[0] / (R * C)] # y[0] is just Q in a different form ... this reads as -Q/(R*C)
sol = solve_ivp(dQdt_sys, [t_start, t_end], y0=[Qo], t_eval=t_num) # (function, time frame, define y0, time array?)
Q_scipy = sol.y[0]

plt.plot(t_num, Q_scipy, label='Scipy')
plt.plot(t_num, Q_true, label='Analytic')
plt.ylabel("Charge (C)")
plt.xlabel("Time (s)")
plt.title("RC Discharge Curve - Scipy")
plt.legend()
plt.show()

plt.plot(t_num, Q_num, label='Euler')
plt.plot(t_rk, Q_rk, label='RK4')
plt.plot(t_num, Q_scipy, label='Scipy')
plt.plot(t_rk, Q_true, label='Analytic')
plt.ylabel("Charge (C)")
plt.xlabel("Time (s)")
plt.title("RC Discharge Curve - All Three")
plt.legend()
plt.show()


# Simple Rienmann Sum
    # chop up R to r_max into a bunch of equal sides rectangles (n points), 
    # evaluate at the left edge og each then multiply by the width and add together

def riemann_sum(f, a, b, n_points, *f_args):
    x = np.linspace(a, b, n_points  + 1)  # edge of each rectangle
    dx = x[1] - x[0]                      # width of each rectangle 

    total = 0.0

    for i in range(n_points):
        total += f(x[i], *f_args) * dx  # height (edge) * width 
    return total

# compare analytical solution to riemann 

W_re_numeric = riemann_sum(integrand, Ri, r_max, n_points, q1, q2, k)
W_true = W_analytic(Ri, q1, q2, k)

print(f"Riemann sum estimate: {W_re_numeric:.6e} J")
print(f"Analytic value:       {W_true:.6e} J")
print(f"Relative error:       {abs(W_re_numeric - W_true)/W_true:.3e}")

# Trapezoidal Rule
    # similar to the Rienmann but uses an adverage of the left and right edge height in place of just the left edge

def trapezoidal_sum(f, a, b, n_points, *f_args):
    x = np.linspace(a, b, n_points  + 1)
    dx = x[1] - x[0] 

    total = 0.0

    for i in range(n_points):
        height = (f(x[i], *f_args) + f(x[i + 1], *f_args)) / 2 # averages the two function values (heights) — it evaluates f twice
        total += height * dx
    return total

# compare analytical solution to trapezoid

W_tr_numeric = trapezoidal_sum(integrand, Ri, r_max, n_points, q1, q2, k)
W_true = W_analytic(Ri, q1, q2, k)

print(f"Trapezoidal sum estimate: {W_tr_numeric:.6e} J")
print(f"Analytic value:       {W_true:.6e} J")
print(f"Relative error:       {abs(W_tr_numeric - W_true)/W_true:.3e}")


# Simpson's Rule
    # fits a parabola through three points at a time which fits the curves better 
    # 3 points are the left edge, right edge, and mid point, fits the parabola to those points then finds the area under the parabola 
    # Needs an even number of intervals 

def simpsons_rule(f, a, b, n_points, *f_args):
    if n_points % 2 != 0:
        n_points += 1  # Simpson's rule needs an even number of intervals

    x = np.linspace(a, b, n_points + 1)
    dx = x[1] - x[0]

    total = f(x[0], *f_args) + f(x[-1], *f_args)  # the two endpoints, weight 1

    for i in range(1, n_points):
        weight = 4 if i % 2 != 0 else 2  # odd index -> 4, even index -> 2
        total += weight * f(x[i], *f_args)

    return (dx / 3) * total

# compare analytical solution to simpsons 

W_si_numeric = simpsons_rule(integrand, Ri, r_max, n_points, q1, q2, k)
W_true = W_analytic(Ri, q1, q2, k)

print(f"Simpsons rule estimate: {W_si_numeric:.6e} J")
print(f"Analytic value:       {W_true:.6e} J")
print(f"Relative error:       {abs(W_si_numeric - W_true)/W_true:.3e}")


# SciPy comparison

def scipy_trapezoidal(f, a, b, n_points, *f_args):
    x = np.linspace(a, b, n_points + 1)
    y = f(x, *f_args)
    return trapezoid(y, x)

def scipy_simpsons(f, a, b, n_points, *f_args):
    x = np.linspace(a, b, n_points + 1)
    y = f(x, *f_args)
    return simpson(y, x)

W_scipy_trap = scipy_trapezoidal(integrand, Ri, r_max, n_points, q1, q2, k)
W_scipy_simp = scipy_simpsons(integrand, Ri, r_max, n_points, q1, q2, k)

print(f"SciPy trapezoidal estimate: {W_scipy_trap:.6e} J")
print(f"Relative error:             {abs(W_scipy_trap - W_true)/W_true:.3e}")

print(f"SciPy Simpson's estimate:   {W_scipy_simp:.6e} J")
print(f"Relative error:             {abs(W_scipy_simp - W_true)/W_true:.3e}")



# Running for errors 

def run_euler(dt, f, *f_args):
    n = int((t_end - t_start) / dt)
    t_local = np.linspace(t_start, t_end, n + 1)
    q = np.zeros(n + 1)
    q[0] = Qo

    for i in range(n):
        q[i + 1] = q[i] + dt * f(t_local[i], q[i], *f_args)

    return q[-1]


def run_RK4(dt, f, *f_args):
    n = int((t_end - t_start) / dt)
    t_local = np.linspace(t_start, t_end, n + 1)
    q = np.zeros(n + 1)
    q[0] = Qo

    for i in range(n):
        h = t_local[i + 1] - t_local[i]
        F1 = h * f(t_local[i], q[i], *f_args)
        F2 = h * f(t_local[i] + h/2, q[i] + F1/2, *f_args)
        F3 = h * f(t_local[i] + h/2, q[i] + F2/2, *f_args)
        F4 = h * f(t_local[i] + h, q[i] + F3, *f_args)
        q[i + 1] = q[i] + (1/6) * (F1 + 2*F2 + 2*F3 + F4)

    return q[-1]


def run_Scipy(dt, f_sys):
    n = int((t_end - t_start) / dt)
    t_local = np.linspace(t_start, t_end, n + 1)
    sol = solve_ivp(f_sys, [t_start, t_end], y0=[Qo], t_eval=t_local)
    return sol.y[0][-1]


dt_values = np.array([0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
q_exact = Qo * np.exp(-t_end / (R * C))

errors_euler = []
errors_RK4 = []
errors_Scipy = []

for dt in dt_values:
    errors_euler.append(abs(run_euler(dt, dQdt, R, C) - q_exact))
    errors_RK4.append(abs(run_RK4(dt, dQdt, R, C) - q_exact))
    errors_Scipy.append(abs(run_Scipy(dt, dQdt_sys) - q_exact))

plt.plot(dt_values, errors_euler, 'o-', label='Euler')
plt.plot(dt_values, errors_RK4, 'o-', label='RK4')
plt.plot(dt_values, errors_Scipy, 'o-', label='Scipy')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time step (dt)')
plt.ylabel('Error at t_max')
plt.title('Change in Error Based on Time Step')
plt.legend()
plt.show()

# Integral Plots 

n_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]  # new list, separate from n_points

riemann_errors = []
trapezoid_errors = []
simpson_errors = []

W_true = W_analytic(Ri, q1, q2, k)

for n in n_values:                     # loop over n_values, not n_points
    W_riemann = riemann_sum(integrand, Ri, r_max, n, q1, q2, k)
    W_trap = trapezoidal_sum(integrand, Ri, r_max, n, q1, q2, k)
    W_simp = simpsons_rule(integrand, Ri, r_max, n, q1, q2, k)

    riemann_errors.append(abs(W_riemann - W_true) / W_true)
    trapezoid_errors.append(abs(W_trap - W_true) / W_true)
    simpson_errors.append(abs(W_simp - W_true) / W_true)

plt.figure(figsize=(7, 5))
plt.loglog(n_values, riemann_errors, "o-", label="Riemann sum")     # n_values here too
plt.loglog(n_values, trapezoid_errors, "o-", label="Trapezoidal")
plt.loglog(n_values, simpson_errors, "o-", label="Simpson's rule")

plt.xlabel("Number of points (n)")
plt.ylabel("Relative error")
plt.title("Convergence of Integration Methods")
plt.legend()
plt.show()

# convergence order as an actual number -- info pulled from the graphing stuff 

log_n = np.log(n_values)            # n_values here too
log_err = np.log(trapezoid_errors)
slope, intercept = np.polyfit(log_n, log_err, 1)
print(f"Trapezoidal convergence order ≈ {-slope:.2f}")


# plots with SciPy

scipy_trap_errors = []
scipy_simp_errors = []

for n in n_values:
    W_scipy_trap = scipy_trapezoidal(integrand, Ri, r_max, n, q1, q2, k)
    W_scipy_simp = scipy_simpsons(integrand, Ri, r_max, n, q1, q2, k)

    scipy_trap_errors.append(abs(W_scipy_trap - W_true) / W_true)
    scipy_simp_errors.append(abs(W_scipy_simp - W_true) / W_true)

plt.figure(figsize=(7, 5))
plt.loglog(n_values, riemann_errors, "o-", label="Riemann sum")
plt.loglog(n_values, trapezoid_errors, "o-", label="Trapezoidal (mine)")
plt.loglog(n_values, simpson_errors, "o-", label="Simpson's (mine)")
plt.loglog(n_values, scipy_trap_errors, "x--", label="Trapezoidal (SciPy)")
plt.loglog(n_values, scipy_simp_errors, "x--", label="Simpson's (SciPy)")

plt.xlabel("Number of points (n)")
plt.ylabel("Relative error")
plt.title("Convergence of Integration Methods")
plt.legend()
plt.show()


#verification stuff

def trap_array(y, x):
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2)

def conservation_check(method, T, n):
    t, Q = method(dQdt, Qo, 0.0, T, n, R, C)
    I = Q / (R * C)                                    # current from the numerical Q(t)

    charge_passed = trap_array(I, t)                   # ∫ I dt
    charge_lost   = Qo - Q[-1]                         # Q0 - Q(T)
    E_diss  = trap_array(I**2 * R, t)                  # ∫ I²R dt
    E_exact = Qo**2 / (2*C) * (1 - np.exp(-2*T/(R*C))) # = 500 J for T >> tau

    print(f"charge: {charge_passed:.8f} vs {charge_lost:.8f}, rel diff {abs(charge_passed-charge_lost)/charge_lost:.2e}")
    print(f"energy: {E_diss:.6f} vs {E_exact:.6f}, rel diff {abs(E_diss-E_exact)/E_exact:.2e}")

for n in (200, 1000, 10000):          # h = 0.05, 0.01, 0.001 at T = 10
    print("n =", n)
    print("Euler"); conservation_check(euler_method, 10.0, n)
    print("RK4");   conservation_check(RK_method,    10.0, n)