"""
Demonstration of signchart with transcendental and composite functions.
"""

import sys

# Force reload of module to get latest changes
if "signchart" in sys.modules:
    del sys.modules["signchart"]
if "signchart.signchart" in sys.modules:
    del sys.modules["signchart.signchart"]

import sympy as sp
from signchart import signchart
import matplotlib.pyplot as plt

x = sp.symbols("x", real=True)

# Example 1: Exponential function
print("Example 1: e^x - 1")
f1 = sp.exp(x) - 1
fig1, ax1 = signchart.plot(f1, domain=(-3, 3), figsize=(10, 3))
plt.savefig("examples/figures/transcendental_exp.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/transcendental_exp.svg", bbox_inches="tight")
print("Saved: transcendental_exp.png/svg\n")

# Example 2: Trigonometric function
print("Example 2: sin(x)")
f2 = sp.sin(x)
fig2, ax2 = signchart.plot(f2, domain=(-sp.pi, sp.pi), figsize=(12, 3))
plt.savefig("examples/figures/transcendental_sin.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/transcendental_sin.svg", bbox_inches="tight")
print("Saved: transcendental_sin.png/svg\n")

# Example 3: Logarithmic function
print("Example 3: ln(x) - 1")
f3 = sp.log(x) - 1
fig3, ax3 = signchart.plot(f3, domain=(0.1, 10), figsize=(10, 3))
plt.savefig("examples/figures/transcendental_log.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/transcendental_log.svg", bbox_inches="tight")
print("Saved: transcendental_log.png/svg\n")

# Example 4: Composite function (polynomial * exponential)
print("Example 4: (x-1)*(x+2)*e^(-x/2)")
f4 = (x - 1) * (x + 2) * sp.exp(-x / 2)
fig4, ax4 = signchart.plot(f4, domain=(-4, 6), figsize=(12, 3))
plt.savefig(
    "examples/figures/transcendental_composite.png", dpi=150, bbox_inches="tight"
)
plt.savefig("examples/figures/transcendental_composite.svg", bbox_inches="tight")
print("Saved: transcendental_composite.png/svg\n")

# Example 5: Product of trig functions
print("Example 5: sin(x)*cos(x)")
f5 = sp.sin(x) * sp.cos(x)
fig5, ax5 = signchart.plot(f5, domain=(-sp.pi, sp.pi), figsize=(14, 3))
plt.savefig(
    "examples/figures/transcendental_trig_product.png", dpi=150, bbox_inches="tight"
)
plt.savefig("examples/figures/transcendental_trig_product.svg", bbox_inches="tight")
print("Saved: transcendental_trig_product.png/svg\n")

print("All transcendental examples generated successfully!")
