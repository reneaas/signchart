# `signchart`
`signchart` is a Python package for plotting sign charts for polynomial functions. It is designed to be simple to use to generate beautiful sign charts for polynomial functions.

## Basic examples

### Example 1

```python
import signchart

f = "(x**2 + 1)**2 * (x - 1)**2 * (x + 1)"

signchart.plot(f=f, include_factors=True)
signchart.savefig(
    dirname="dirname",
    fname="fname",
)

signchart.show()
```

This will generate the following sign chart:

![sign chart](https://raw.githubusercontent.com/reneaas/signchart/refs/heads/main/examples/figures/example_1.svg)


### Example 2

```python
import signchart

f = "x**2 - x - 6"

signchart.plot(
    f=f,
    include_factors=True,
    color=True,  # Includes colored lines.
    fn_name="g(x)",  # Names the function g(x)
)

signchart.savefig(
    dirname="figures",
    fname="example_2.svg",
)

signchart.show()
```

This will generate the following sign chart:

![sign chart](https://raw.githubusercontent.com/reneaas/signchart/refs/heads/main/examples/figures/example_2.svg)

### Example 3

```python
import signchart

f = "-2 * x**2 + 2 * x + 12"

signchart.plot(
    f=f,
    include_factors=True,
    color=True,  # Includes colored lines.
    fn_name="h(x)",  # Names the function h(x)
)

signchart.savefig(
    dirname="figures",
    fname="example_3.svg",
)

signchart.show()
```

This will generate the following sign chart:

![sign chart](https://raw.githubusercontent.com/reneaas/signchart/refs/heads/main/examples/figures/example_3.svg)


### Example 4

```python
import signchart

f = "-3 * (t - 1) * (t + 3)"  # Uses 't' as variable in place of 'x'

signchart.plot(
    f=f,
    include_factors=False,  # excludes linear factors in the polynomial
    color=False,  # sign lines are black (uncolored)
    fn_name="x(t)",  # Names the function x(t)
)

signchart.savefig(
    dirname="figures",
    fname="example_4.svg",
)

signchart.show()
```

This will generate the following sign chart:

![sign chart](https://raw.githubusercontent.com/reneaas/signchart/refs/heads/main/examples/figures/example_4.svg)
