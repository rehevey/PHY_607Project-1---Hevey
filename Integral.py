import numpy as np
from scipy.integrate import trapezoid
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Defining our variables 

k = 8.9875e9 # Coulomb's constant 
q1 = 1.0e-6  # charge 1 in Coulombs 
q2 = 1.0e-6 #charge 2 in Coulombs 
R = 0.01 # lower bound (starting seperation), meters 
r_max = 10.0 # truncated stand-in for infinity, meters 
n_points = 5000 

# Analytical Solution 

def integrand (r, q1, q2, k):
    return (k * q1 * q2) / (r**2)

def W_analytic(R, q1, q2, k):
    return (k * q1 * q2) / R 

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

W_re_numeric = riemann_sum(integrand, R, r_max, n_points, q1, q2, k)
W_true = W_analytic(R, q1, q2, k)

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

W_tr_numeric = trapezoidal_sum(integrand, R, r_max, n_points, q1, q2, k)
W_true = W_analytic(R, q1, q2, k)

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

W_si_numeric = simpsons_rule(integrand, R, r_max, n_points, q1, q2, k)
W_true = W_analytic(R, q1, q2, k)

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

W_scipy_trap = scipy_trapezoidal(integrand, R, r_max, n_points, q1, q2, k)
W_scipy_simp = scipy_simpsons(integrand, R, r_max, n_points, q1, q2, k)

print(f"SciPy trapezoidal estimate: {W_scipy_trap:.6e} J")
print(f"Relative error:             {abs(W_scipy_trap - W_true)/W_true:.3e}")

print(f"SciPy Simpson's estimate:   {W_scipy_simp:.6e} J")
print(f"Relative error:             {abs(W_scipy_simp - W_true)/W_true:.3e}")


# Integral Plots 

n_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]  # new list, separate from n_points

riemann_errors = []
trapezoid_errors = []
simpson_errors = []

W_true = W_analytic(R, q1, q2, k)

for n in n_values:                     # loop over n_values, not n_points
    W_riemann = riemann_sum(integrand, R, r_max, n, q1, q2, k)
    W_trap = trapezoidal_sum(integrand, R, r_max, n, q1, q2, k)
    W_simp = simpsons_rule(integrand, R, r_max, n, q1, q2, k)

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
    W_scipy_trap = scipy_trapezoidal(integrand, R, r_max, n, q1, q2, k)
    W_scipy_simp = scipy_simpsons(integrand, R, r_max, n, q1, q2, k)

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
