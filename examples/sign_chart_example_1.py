from signchart import make_sign_chart, show, savefig

f = "(x**2 + 1)**2 * (x - 1)**2 * (x + 1)"
make_sign_chart(f=f, include_factors=True)
show()

savefig(
    dirname="figures",
    fname="example_1.svg",
)
