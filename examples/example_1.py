import signchart

f = "(x**2 + 1)**2 * (x - 1)**2 * (x + 1)"

signchart.plot(f=f, include_factors=True)
signchart.savefig(
    dirname="figures",
    fname="example_1.png",
)

signchart.show()
