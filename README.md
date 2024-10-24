# `signchart`
`signchart` is a Python package for plotting sign charts for polynomial functions. It is designed to be simple to use to generate beatiful sign charts for polynomial functions.

Example usage:

```python
from signchart import make_sign_chart, show

f = "(x**2 + 1)**2 * (x - 1)**2 * (x + 1)"
make_sign_chart(f=f, include_factors=True)
show()
```