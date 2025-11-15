"""
Test: Sign chart with a factor that has no zeros (exp(x²))

This demonstrates that factors without zeros are now included
in the sign chart and shown as constant positive or negative lines.
"""

from signchart import plot, savefig
import sympy as sp

x = sp.Symbol("x")
f = (x**2 - 1) * sp.exp(x**2)

plot(f, x, domain=(-3, 3))
savefig("figures", "test_exp_factor.png")
