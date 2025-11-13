# Quick Reference: Transcendental Functions in Signchart

## Basic Usage

### Polynomial Functions (No domain needed)
```python
import sympy as sp
from signchart import signchart

x = sp.symbols('x', real=True)

# Polynomials work automatically
f = x**2 - 4
signchart.plot(f)
```

### Transcendental Functions (Domain required)
```python
# Exponential: e^x - 1
f = sp.exp(x) - 1
signchart.plot(f, domain=(-3, 3))

# Trigonometric: sin(x)
f = sp.sin(x)
signchart.plot(f, domain=(-sp.pi, sp.pi))

# Logarithmic: ln(x) - 1
f = sp.log(x) - 1
signchart.plot(f, domain=(0.1, 10))
```

## Common Patterns

### Periodic Functions
```python
# One period of sine
signchart.plot(sp.sin(x), domain=(0, 2*sp.pi))

# Multiple periods
signchart.plot(sp.cos(x), domain=(-2*sp.pi, 2*sp.pi))
```

### Exponential Decay/Growth
```python
# Decay: e^(-x)
signchart.plot(sp.exp(-x), domain=(-2, 5))

# Growth: e^x - 2
signchart.plot(sp.exp(x) - 2, domain=(-3, 3))
```

### Logarithmic Functions
```python
# Natural log
signchart.plot(sp.log(x), domain=(0.01, 10))

# Shifted log: ln(x-1)
signchart.plot(sp.log(x - 1), domain=(1.1, 10))
```

### Composite Functions
```python
# Polynomial × Exponential
f = (x - 1) * (x + 2) * sp.exp(-x/2)
signchart.plot(f, domain=(-4, 6))

# Trig × Polynomial
f = sp.sin(x) * (x**2 - 4)
signchart.plot(f, domain=(-sp.pi, sp.pi))
```

## Important Notes

1. **Domain is required** for non-polynomial functions
2. **Choose appropriate domain**:
   - For trig: use multiples of π
   - For exp: consider asymptotic behavior
   - For log: domain must be > 0
3. **Numerical accuracy**: Zeros found to ~1e-10 precision
4. **Sampling**: 1000 points sampled across domain

## Troubleshooting

### Missing zeros?
- Expand the domain
- Check for rapid oscillations
- Ensure domain includes all zeros of interest

### Singularity confusion?
- The module automatically filters singularities from zeros
- For 1/x: singularity at 0, no zeros (correct)
- For (x-2)/(x+1): zero at 2, singularity at -1 (correct)

### Performance?
- Narrower domains = faster computation
- Polynomial/rational functions don't need domain (faster)
