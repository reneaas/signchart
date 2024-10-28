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
