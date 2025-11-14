"""
Demonstration of factor display for transcendental functions.

This shows that the signchart module can now display individual factors
for transcendental and composite functions, not just polynomials.
"""

import sympy as sp
from signchart import signchart
import matplotlib.pyplot as plt

x = sp.symbols("x", real=True)

print("=" * 80)
print("TRANSCENDENTAL FACTOR DISPLAY DEMO")
print("=" * 80)

# Example 1: Product of trigonometric functions
print("\n1. Product: sin(x) * cos(x)")
print("   Expected factors: sin(x) and cos(x)")
f1 = sp.sin(x) * sp.cos(x)
fig1, ax1 = signchart.plot(f1, domain=(-sp.pi, sp.pi), figsize=(14, 5))
plt.savefig("examples/figures/demo_sin_cos_factors.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/demo_sin_cos_factors.svg", bbox_inches="tight")
plt.close(fig1)
print("   ✓ Saved")

# Example 2: Polynomial * Transcendental
print("\n2. Composite: (x² - 4) * sin(x)")
print("   Expected factors: (x-2), (x+2), and sin(x)")
f2 = (x**2 - 4) * sp.sin(x)
fig2, ax2 = signchart.plot(f2, domain=(-4, 4), figsize=(16, 6))
plt.savefig("examples/figures/demo_poly_sin_factors.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/demo_poly_sin_factors.svg", bbox_inches="tight")
plt.close(fig2)
print("   ✓ Saved")

# Example 3: Polynomial * Exponential
print("\n3. Composite: x * e^(-x)")
print("   Expected factors: x and e^(-x)")
f3 = x * sp.exp(-x)
fig3, ax3 = signchart.plot(f3, domain=(-2, 5), figsize=(12, 4))
plt.savefig("examples/figures/demo_x_exp_factors.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/demo_x_exp_factors.svg", bbox_inches="tight")
plt.close(fig3)
print("   ✓ Saved")

# Example 4: Rational transcendental
print("\n4. Rational: sin(x) / cos(x) = tan(x)")
print("   Expected factors: sin(x) in numerator, cos(x) in denominator")
f4 = sp.sin(x) / sp.cos(x)
fig4, ax4 = signchart.plot(f4, domain=(-1.4, 1.4), figsize=(12, 4))
plt.savefig("examples/figures/demo_tan_factors.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/demo_tan_factors.svg", bbox_inches="tight")
plt.close(fig4)
print("   ✓ Saved")

# Example 5: Complex product
print("\n5. Complex: (x² - 1) * sin(x) * cos(x)")
print("   Expected factors: (x-1), (x+1), sin(x), cos(x)")
f5 = (x**2 - 1) * sp.sin(x) * sp.cos(x)
fig5, ax5 = signchart.plot(f5, domain=(-sp.pi, sp.pi), figsize=(16, 7))
plt.savefig("examples/figures/demo_complex_factors.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/demo_complex_factors.svg", bbox_inches="tight")
plt.close(fig5)
print("   ✓ Saved")

print("\n" + "=" * 80)
print("DEMO COMPLETED")
print("=" * 80)
print("\nGenerated files:")
print("  • demo_sin_cos_factors.png/svg")
print("  • demo_poly_sin_factors.png/svg")
print("  • demo_x_exp_factors.png/svg")
print("  • demo_tan_factors.png/svg")
print("  • demo_complex_factors.png/svg")
print("\nThe sign charts now show individual factor rows for transcendental functions!")
print("Each factor (sin, cos, exp, etc.) is displayed on its own row with its zeros.")
