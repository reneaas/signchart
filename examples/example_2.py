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
