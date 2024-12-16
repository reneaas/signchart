import signchart


f = "-3*(x**2 - x - 2) / ((x - 1) * (x + 2))"

signchart.plot(
    f=f,
    include_factors=True,
)


signchart.show()
