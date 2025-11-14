"""
Example: Sign chart for f(x) = cos³(x)

This demonstrates that power expressions of transcendental functions
are now correctly handled, showing:
- Single factor cos(x) with exponent 3
- Correct sign pattern: negative → positive → negative
"""

from signchart import plot, savefig
import sympy as sp

x = sp.Symbol("x")
f = sp.cos(x) ** 3

plot(f, x, domain=(-2 * sp.pi, 2 * sp.pi))
savefig("figures", "example_cos_cubed.png")
