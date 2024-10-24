import signchart

g = "x**2 - x - 6"

signchart.make_sign_chart(
    f=g,
    include_factors=True,
    color=True,  # Includes colored lines.
    fn_name="g(x)",  # Names the function g(x)
)

signchart.savefig(
    dirname="figures",
    fname="example_2.svg",
)

signchart.show()
