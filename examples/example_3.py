import signchart

g = "-2 * x**2 + 2 * x + 12"

signchart.make_sign_chart(
    f=g,
    include_factors=True,
    color=True,  # Includes colored lines.
    fn_name="g(x)",  # Names the function g(x)
)

signchart.savefig(
    dirname="figures",
    fname="example_3.svg",
)

signchart.show()
