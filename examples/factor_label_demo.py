"""
Demonstration of proper LaTeX rendering for factor labels.
This shows that expressions like x - (1 + 2√5) are now rendered correctly.
"""

import sympy as sp
from signchart import signchart
import matplotlib.pyplot as plt

x = sp.symbols("x", real=True)

# Example with complex irrational roots
print("Demonstrating LaTeX rendering for factor labels")
print("=" * 60)

# Function with roots at 1 ± 2√5
f = x**2 - 2 * x - 19

roots = sp.solve(f, x)
print(f"\nFunction: f(x) = {f}")
print(f"Roots: {roots}")
print(f"\nRoot 1: {roots[0]}")
print(f"  LaTeX: {sp.latex(roots[0])}")
print(f"  Factor: x - ({roots[0]})")
print(f"  Factor LaTeX: {sp.latex(sp.simplify(x - roots[0]))}")

print(f"\nRoot 2: {roots[1]}")
print(f"  LaTeX: {sp.latex(roots[1])}")
print(f"  Factor: x - ({roots[1]})")
print(f"  Factor LaTeX: {sp.latex(sp.simplify(x - roots[1]))}")

print("\n" + "=" * 60)
print("Creating sign chart...")
print("=" * 60)

fig, ax = signchart.plot(f, figsize=(12, 4))
plt.savefig("examples/figures/factor_label_demo.png", dpi=150, bbox_inches="tight")
plt.savefig("examples/figures/factor_label_demo.svg", bbox_inches="tight")

print("\n✓ Sign chart saved to:")
print("  - examples/figures/factor_label_demo.png")
print("  - examples/figures/factor_label_demo.svg")
print("\nThe LEFT SIDE labels should render as proper LaTeX:")
print("  • x - 1 + 2√5  (for root 1 - 2√5)")
print("  • x - 2√5 - 1  (for root 1 + 2√5)")
print("\nNOT as plain text:")
print("  • x - 1 + 2sqrt(5)  ✗ WRONG (old behavior)")
