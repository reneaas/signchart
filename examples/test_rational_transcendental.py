"""
Test: Sign chart for rational function with transcendental denominator

This demonstrates that denominator factors are now properly extracted
and shown in the sign chart, with singularities marked.
"""

from signchart import plot, savefig
import sympy as sp

x = sp.Symbol("x")
f = (x**2 - 1) / (sp.exp(x) - 1)

plot(f, x, domain=(-3, 3))
savefig("figures", "test_rational_transcendental.png")
