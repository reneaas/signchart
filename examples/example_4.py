import signchart

f = "-3 * (t - 1) * (t + 3)"  # Uses 't' as variable in place of 'x'

signchart.make_sign_chart(
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
