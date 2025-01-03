import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pathlib

import warnings

# Attempt to enable LaTeX rendering
try:
    plt.rc("text", usetex=True)
    # Test rendering a simple LaTeX expression
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, r"$E=mc^2$", fontsize=12)
    plt.close(fig)  # Close the figure as it's only for testing
except Exception:
    plt.rc("text", usetex=False)
    warnings.warn(
        "LaTeX is not available. Falling back to Matplotlib's default text rendering.",
        UserWarning,
    )


def savefig(dirname, fname):
    dir = pathlib.Path(dirname)
    dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{dir}/{fname}")

    return None


def show():
    plt.show()


def get_factors(polynomial, x):
    polynomial = sp.expand(polynomial)
    factor_list = sp.factor_list(polynomial)
    leading_coeff = factor_list[0]

    linear_factors = (
        [{"expression": leading_coeff, "exponent": 1, "root": -np.inf}]
        if leading_coeff != 1
        else []
    )

    for linear_factor, exponent in factor_list[1]:
        exponent = int(exponent)
        root = sp.solve(linear_factor, x)
        root_value = root[0] if root else -np.inf
        linear_factors.append(
            {
                "expression": linear_factor,
                "exponent": exponent,
                "root": root_value,
            }
        )

    return linear_factors


def sort_factors(factors):
    factors = sorted(factors, key=lambda x: x.get("root"))
    return factors


def draw_factors(
    f,
    factors,
    roots,
    root_positions,
    ax,
    color_pos,
    color_neg,
    x,
    dy=-1,
    dx=0.02,
):
    x_min = -0.05
    x_max = 1.05
    # Draw horizontal sign lines for each factor
    for i, factor in enumerate(factors):
        expression = str(factor.get("expression"))
        exponent = factor.get("exponent")

        expression = f"${sp.latex(sp.sympify(expression))}$"
        s = expression
        # Replace ** with ^ for sympy expressions to work with latex
        # if "**" in str(expression):
        #     expression = expression.replace("**", "^")

        # # Remove multiplication signs
        # if "*" in str(expression):
        #     expression = expression.replace("*", "")

        # if exponent > 1:
        #     s = f"$({expression})^{exponent}$"
        # else:
        #     s = f"${expression}$"

        plt.text(
            x=-0.1,
            y=(i + 1) * dy,
            s=s,
            fontsize=16,
            ha="right",
            va="center",
        )
        if factor.get("root") == -np.inf:
            y_value = sp.sympify(factor.get("expression")).evalf(subs={x: 0})
            if y_value > 0:
                ax.plot(
                    [x_min, x_max],
                    [(i + 1) * dy, (i + 1) * dy],
                    color=color_pos,
                    linestyle="-",
                    lw=2,
                )
            else:
                ax.plot(
                    [x_min, x_max],
                    [(i + 1) * dy, (i + 1) * dy],
                    color=color_neg,
                    linestyle="--",
                    lw=2,
                )
        elif factor.get("exponent") % 2 == 0:
            root = factor.get("root")
            root_pos = root_positions[root]

            ax.plot(
                [x_min, root_pos - dx],
                [(i + 1) * dy, (i + 1) * dy],
                color=color_pos,
                linestyle="-",
                lw=2,
            )
            ax.plot(
                [root_pos + dx, x_max],
                [(i + 1) * dy, (i + 1) * dy],
                color=color_pos,
                linestyle="-",
                lw=2,
            )

            if str(f.subs(x, root)) != "zoo":
                plt.text(
                    x=root_pos,
                    y=(i + 1) * dy,
                    s=f"$0$",
                    fontsize=20,
                    ha="center",
                    va="center",
                )
            else:
                plt.text(
                    x=root_pos + 0.005,
                    y=(i + 1) * dy,
                    s=f"$\\times$",
                    fontsize=24,
                    ha="center",
                    va="center",
                )
        else:
            root = factor.get("root")
            root_pos = root_positions[root]

            ax.plot(
                [x_min, root_pos - dx],
                [(i + 1) * dy, (i + 1) * dy],
                color=color_neg,
                linestyle="--",
                lw=2,
            )
            ax.plot(
                [root_pos + dx, x_max],
                [(i + 1) * dy, (i + 1) * dy],
                color=color_pos,
                linestyle="-",
                lw=2,
            )

            if str(f.subs(x, root)) != "zoo":
                plt.text(
                    x=root_pos,
                    y=(i + 1) * dy,
                    s=f"$0$",
                    fontsize=20,
                    ha="center",
                    va="center",
                )
            else:
                plt.text(
                    x=root_pos + 0.005,
                    y=(i + 1) * dy,
                    s=f"$\\times$",
                    fontsize=24,
                    ha="center",
                    va="center",
                )


def draw_function(
    factors,
    roots,
    root_positions,
    ax,
    color_pos,
    color_neg,
    x,
    f,
    fn_name=None,
    include_factors=True,
    dy=-1,
    dx=0.02,
):

    # roots = [factor.get("root") for factor in factors]
    # roots.remove(-np.inf)
    # print(roots)
    x_min = -0.05
    x_max = 1.05

    if include_factors:
        y = (len(factors) + 1) * dy
    else:
        y = dy
    plt.text(
        x=-0.1,
        y=y,
        s=f"${fn_name}$" if fn_name else f"$f({str(x)})$",
        fontsize=16,
        ha="right",
        va="center",
    )

    # Case 1: the polynomial has no roots.
    if len(roots) == 0:
        x0 = 0
        y0 = sp.sympify(f).evalf(subs={x: x0})

        if y0 > 0:
            ax.plot(
                [x_min, x_max],
                [y, y],
                color=color_pos,
                linestyle="-",
                lw=2,
            )
        else:
            ax.plot(
                [x_min, x_max],
                [y, y],
                color=color_neg,
                linestyle="--",
                lw=2,
            )

        return None

    intervals = []
    interval_positions = []

    # Intervals before first root
    intervals.append((roots[0] - 1, roots[0] - 0.1))
    interval_positions.append((x_min, root_positions[roots[0]] - dx))

    # Intervals between roots
    for i in range(len(roots) - 1):
        intervals.append((roots[i] + 0.1, roots[i + 1] - 0.1))
        interval_positions.append(
            (root_positions[roots[i]] + dx, root_positions[roots[i + 1]] - dx)
        )

    # Interval after last root
    intervals.append((roots[-1] + 0.1, roots[-1] + 1))
    interval_positions.append((root_positions[roots[-1]] + dx, x_max))

    for i, (x0_interval, pos_interval) in enumerate(zip(intervals, interval_positions)):
        x0 = (x0_interval[0] + x0_interval[1]) / 2
        y0 = sp.sympify(f).evalf(subs={x: x0})

        if y0 > 0:
            ax.plot(
                [pos_interval[0], pos_interval[1]],
                [y, y],
                color=color_pos,
                linestyle="-",
                lw=2,
            )
        else:
            ax.plot(
                [pos_interval[0], pos_interval[1]],
                [y, y],
                color=color_neg,
                linestyle="--",
                lw=2,
            )

    # Plot zeros at root positions
    for root in roots:
        root_pos = root_positions[root]
        if str(f.subs(x, root)) != "zoo":
            plt.text(
                x=root_pos,
                y=y,
                s=f"$0$",
                fontsize=20,
                ha="center",
                va="center",
            )
        else:
            plt.text(
                x=root_pos + 0.005,
                y=y,
                s=f"$\\times$",
                fontsize=24,
                ha="center",
                va="center",
            )


def draw_vertical_lines(
    roots,
    root_positions,
    factors,
    ax,
    include_factors=True,
    dy=-1,
):
    # Draw vertical lines to separate regions
    offset_dy = 0.2

    if include_factors:
        # Collect y positions of zeros from factors
        y_zeros_dict = {}
        for i, factor in enumerate(factors):
            if factor.get("root") != -np.inf:
                root = factor.get("root")
                y_zero = (i + 1) * dy
                if root in y_zeros_dict:
                    y_zeros_dict[root].append(y_zero)
                else:
                    y_zeros_dict[root] = [y_zero]
        # Add y position of zero from function
        y_function = (len(factors) + 1) * dy
    else:
        y_zeros_dict = {}
        y_function = dy

    y_min = -0.4
    y_max = y_function + 0.5

    for root in roots:
        root_pos = root_positions[root]
        # Collect y positions where zeros are placed at this root
        zero_y_positions = []
        # From factors
        if root in y_zeros_dict:
            zero_y_positions.extend(y_zeros_dict[root])
        # From function
        zero_y_positions.append(y_function)
        # Now adjust zero_y_positions to include offset_dy
        y_positions = [y_min]
        for y_zero in zero_y_positions:
            y_positions.extend([y_zero - offset_dy, y_zero + offset_dy])
        y_positions.append(y_max)
        y_positions = sorted(y_positions)

        # Now plot segments between pairs
        for i in range(1, len(y_positions) - 1):
            y_start = y_positions[i]
            y_end = y_positions[i + 1]
            # Skip the segments around the zeros
            if (i % 2) == 0:
                if y_end - y_start > 0:
                    ax.plot(
                        [root_pos, root_pos],
                        [y_start, y_end],
                        color="black",
                        linestyle="-",
                        lw=1,
                    )


def make_axis(x):
    fig, ax = plt.subplots()

    # Remove y-axis spines
    ax.spines["left"].set_color("none")  # Remove the left y-axis
    ax.spines["right"].set_color("none")  # Remove the right y-axis

    # Move the x-axis to y=0
    ax.spines["bottom"].set_position("zero")  # Position the bottom x-axis at y=0
    ax.spines["top"].set_color("none")  # Remove the top x-axis

    # Move x-axis ticks and labels to the top
    ax.xaxis.set_ticks_position("top")  # Move ticks to the top
    ax.xaxis.set_label_position("top")  # Move labels to the top
    ax.tick_params(
        axis="x",
        which="both",  # Hide bottom ticks and labels
        bottom=False,
        labelbottom=False,
        length=10,
    )

    # Attach arrow to the right end of the x-axis
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)

    # Label the x-axis
    ax.set_xlabel(f"${str(x)}$", fontsize=16, loc="right")

    # Remove tick labels on y-axis
    plt.yticks([])

    # Set x-limits
    ax.set_xlim(-0.05, 1.05)

    return fig, ax


def plot(
    f,
    x=None,
    fn_name=None,
    color=True,
    include_factors=True,
    generic_labels=False,
    small_figsize=False,
):
    """Draws a sign chart for a polynomial f.

    Args:
        f (sp.Expr, str):
            Polynomial. May be a sympy.Expr or str.
        x (sp.symbols, str, optional):
            Variable in the polynomial
        fn_name (str, optional):
            Name of the function. Defaults `None`.
        color (bool, optional):
            Enables coloring of sign chart. Default: `True`.
        include_factors (bool, optional):
            Includes all linear factors of f(x). Default: `True`.
        generic_label (bool, optional):
            Uses generic labels for roots: x_1, x_2, ..., x_N. Default: `False`.
        small_figsize (bool, optional):
            Enables rescaling of the figure for a smaller figure size. Default: `False`.

    Returns:
        fig (plt.figure)
            matplotlib figure.
        ax (plt.Axis)
            matplotlib axis.
    """
    if isinstance(f, str):
        f = sp.sympify(f)

    original_variable = list(f.free_symbols)[0]
    x = sp.symbols(str(original_variable), real=True)
    f = f.subs(original_variable, x)

    if color:
        color_pos = "red"
        color_neg = "blue"
    else:
        color_pos = color_neg = "black"

    if not f.is_polynomial():
        # Assume a rational function
        p, q = str(f).split("/")
        p = sp.sympify(p)
        q = sp.sympify(q)

        original_variable = list(p.free_symbols)[0]
        p = p.subs(original_variable, x)

        original_variable = list(q.free_symbols)[0]
        q = q.subs(original_variable, x)

        # Order the factors
        p_factors = get_factors(polynomial=p, x=x)

        q_factors = get_factors(polynomial=q, x=x)

        factors = p_factors + q_factors

        factors = sort_factors(factors)

    else:
        factors = get_factors(polynomial=f, x=x)  # compute linear factors
        factors = sort_factors(
            factors=factors
        )  # Sort linear factors in ascending order.

    print(f"Creating sign chart for f(x) = {f} = {f.factor()}")

    # Create figure
    fig, ax = make_axis(x)

    # Extract roots
    roots = [factor.get("root") for factor in factors if factor.get("root") != -np.inf]

    # Map roots to positions
    num_roots = len(roots)
    x_min = -0.05
    x_max = 1.05
    positions = np.linspace(0, 1, num_roots + 2)[1:-1]  # Exclude 0 and 1
    root_positions = dict(zip(roots, positions))

    # Set tick marks for roots of the polynomial
    plt.xticks(
        ticks=positions,
        labels=[
            f"${sp.latex(sp.sympify(root))}$" if not generic_labels else f"$x_{i + 1}$"
            for i, root in enumerate(roots)
        ],
        fontsize=16,
    )

    # Draw factors
    if include_factors:
        draw_factors(f, factors, roots, root_positions, ax, color_pos, color_neg, x)

    # Draw sign lines for function
    draw_function(
        factors,
        roots,
        root_positions,
        ax,
        color_pos,
        color_neg,
        x,
        f,
        fn_name,
        include_factors,
    )

    # Remove tick labels on y-axis
    plt.yticks([])

    plt.xlim(x_min, x_max)

    if include_factors:
        fig.set_size_inches(8, 2 + int(0.7 * len(factors)))

    elif small_figsize:
        fig.set_size_inches(4, 1.5)
    else:
        fig.set_size_inches(8, 2)

    draw_vertical_lines(roots, root_positions, factors, ax, include_factors)

    plt.tight_layout()

    return fig, ax
