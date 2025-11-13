# Transcendental Function Support

## Overview

The `signchart` module now supports arbitrary composite functions including transcendental functions such as:
- Exponential functions (`exp`, `e^x`)
- Logarithmic functions (`log`, `ln`)
- Trigonometric functions (`sin`, `cos`, `tan`, etc.)
- Hyperbolic functions (`sinh`, `cosh`, `tanh`, etc.)
- Composite functions (combinations of the above with polynomials)

## New Features

### 1. Domain Parameter

For non-polynomial functions, you must specify a domain over which to search for zeros:

```python
import sympy as sp
from signchart import signchart

x = sp.symbols('x', real=True)

# Exponential function
f = sp.exp(x) - 1
fig, ax = signchart.plot(f, domain=(-3, 3))

# Trigonometric function
f = sp.sin(x)
fig, ax = signchart.plot(f, domain=(-sp.pi, sp.pi))

# Logarithmic function
f = sp.log(x) - 1
fig, ax = signchart.plot(f, domain=(0.1, 10))
```

### 2. Automatic Zero Detection

The module uses a hybrid approach to find zeros:

1. **Symbolic solving**: Attempts to find exact zeros using SymPy's `solve()` function
2. **Numerical sampling**: Samples 1000 points across the domain to detect sign changes
3. **Root refinement**: Uses Brent's method to refine numerical zeros to high precision

### 3. Singularity Detection

For rational functions, the module automatically:
- Detects singularities (poles) from the denominator
- Excludes singularities from the zero list
- Properly handles discontinuities

## Examples

### Exponential Functions

```python
# e^x - 1 has a zero at x = 0
f = sp.exp(x) - 1
signchart.plot(f, domain=(-3, 3))
```

### Trigonometric Functions

```python
# sin(x) has zeros at -π, 0, π
f = sp.sin(x)
signchart.plot(f, domain=(-sp.pi, sp.pi))

# Product of trig functions
f = sp.sin(x) * sp.cos(x)
signchart.plot(f, domain=(-sp.pi, sp.pi))
```

### Logarithmic Functions

```python
# ln(x) - 1 has a zero at x = e
f = sp.log(x) - 1
signchart.plot(f, domain=(0.1, 10))
```

### Composite Functions

```python
# Polynomial * Exponential
f = (x - 1) * (x + 2) * sp.exp(-x/2)
signchart.plot(f, domain=(-4, 6))

# Trigonometric polynomial
f = sp.sin(x) * (x**2 - 4)
signchart.plot(f, domain=(-sp.pi, sp.pi))
```

## Technical Details

### Zero Finding Algorithm

For transcendental functions, the `get_zeros_and_singularities()` function:

1. **Detects singularities first** by analyzing the denominator
2. **Symbolic zeros**: Attempts `sp.solve(f, x)` for exact solutions
3. **Numerical zeros**: 
   - Samples 1000 points in the domain
   - Excludes points near singularities (within 1e-4)
   - Detects sign changes: `f(x[i]) * f(x[i+1]) < 0`
   - Refines zeros using `scipy.optimize.brentq`
4. **Filters results**: Removes any zeros within 1e-6 of singularities

### Limitations

- **Domain required**: You must specify a domain for transcendental functions
- **Oscillatory functions**: May miss zeros in regions with rapid oscillations
- **Numerical precision**: Zeros are accurate to approximately 1e-10
- **Multi-valued functions**: Does not handle branch cuts (e.g., complex logarithm)

## Migration from Polynomial-Only Usage

### Before (Polynomials only)
```python
f = x**2 - 4
signchart.plot(f)  # Works automatically
```

### After (Still works the same)
```python
f = x**2 - 4
signchart.plot(f)  # Still works without domain
```

### New (Transcendental functions)
```python
f = sp.sin(x)
signchart.plot(f, domain=(-sp.pi, sp.pi))  # Domain required
```

## Performance Considerations

- **Polynomial/Rational functions**: Fast symbolic factorization (no domain needed)
- **Transcendental functions**: Requires numerical sampling (1000 points) and root refinement
- **Large domains**: Use narrower domains for faster computation
- **Periodic functions**: Choose domain to capture desired period

## Testing

Run the comprehensive test suite:

```bash
python examples/transcendental_demo.py
```

This generates sign charts for:
- Exponential: `e^x - 1`
- Trigonometric: `sin(x)`, `sin(x)*cos(x)`
- Logarithmic: `ln(x) - 1`
- Composite: `(x-1)*(x+2)*e^(-x/2)`

## See Also

- Main examples: `examples/example_*.py`
- Transcendental examples: `examples/transcendental_demo.py`
- Generated figures: `examples/figures/transcendental_*.png`
