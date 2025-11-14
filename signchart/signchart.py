import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pathlib
import shutil
import platform

import warnings

# Check if LaTeX is available
if platform.system() == "Windows":
    latex_available = shutil.which("latex.exe") is not None
else:
    latex_available = shutil.which("latex") is not None


if latex_available:
    try:
        plt.rc("text", usetex=True)
    except (FileNotFoundError, RuntimeError):
        plt.rc("text", usetex=False)
else:
    plt.rc("text", usetex=False)


def savefig(dirname, fname):
    import plotmath

    plotmath.savefig(dirname, fname)

    return None


def show():
    plt.show()


def get_zeros_and_singularities(f, x, domain=None):
    """Find zeros and singularities (poles/discontinuities) of arbitrary functions.

    Args:
        f: sympy expression
        x: variable symbol
        domain: optional tuple (x_min, x_max) to search for numerical zeros

    Returns:
        dict with 'zeros' and 'singularities' lists containing symbolic/numeric values
    """
    zeros = []
    singularities = []

    # Detect singularities FIRST (for rational/transcendental functions)
    try:
        # Check denominator for rational functions
        numer, denom = f.as_numer_denom()
        if denom != 1:
            denom_zeros = sp.solve(denom, x, domain=sp.S.Reals)
            for z in denom_zeros:
                if z.is_real or (hasattr(z, "is_real") and z.is_real is not False):
                    singularities.append(z)
    except:
        pass

    # Try symbolic solving for zeros
    try:
        symbolic_zeros = sp.solve(f, x, domain=sp.S.Reals)
        if symbolic_zeros:
            for z in symbolic_zeros:
                if z.is_real or (hasattr(z, "is_real") and z.is_real is not False):
                    # Make sure this isn't a singularity
                    is_sing = False
                    for sing in singularities:
                        try:
                            if abs(float((z - sing).evalf())) < 1e-10:
                                is_sing = True
                                break
                        except:
                            pass
                    if not is_sing:
                        zeros.append(z)
    except (NotImplementedError, ValueError):
        pass

    # For transcendental/complex functions, find numerical zeros if domain provided
    if domain:
        try:
            # Sample points to find sign changes
            x_min, x_max = domain
            test_points = np.linspace(float(x_min), float(x_max), 1000)
            f_lamb = sp.lambdify(x, f, modules=["numpy"])

            # Suppress warnings for evaluation at singularities
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_vals = []
                valid_x = []
                for xp in test_points:
                    try:
                        yp = float(f_lamb(xp))
                        # Exclude points near singularities
                        near_sing = False
                        for sing in singularities:
                            try:
                                if abs(xp - float(sing.evalf())) < 1e-4:
                                    near_sing = True
                                    break
                            except:
                                pass
                        if not near_sing and np.isfinite(yp) and abs(yp) < 1e10:
                            y_vals.append(yp)
                            valid_x.append(xp)
                    except:
                        pass

                y_vals = np.array(y_vals)
                valid_x = np.array(valid_x)

                # Find sign changes (potential zeros) and near-zero values
                numerical_zeros = []
                if len(y_vals) > 1:
                    # Check for sign changes
                    signs = np.sign(y_vals)
                    sign_changes = np.where(np.diff(signs) != 0)[0]

                    for idx in sign_changes:
                        # Skip if both values are zero (avoid duplicates)
                        if abs(y_vals[idx]) < 1e-10 and abs(y_vals[idx + 1]) < 1e-10:
                            continue
                        # Refine zero location
                        try:
                            from scipy.optimize import brentq

                            x_zero = brentq(
                                f_lamb, valid_x[idx], valid_x[idx + 1], xtol=1e-10
                            )
                            numerical_zeros.append(sp.Float(x_zero))
                        except:
                            # Use midpoint if brentq fails
                            x_zero = (valid_x[idx] + valid_x[idx + 1]) / 2
                            numerical_zeros.append(sp.Float(x_zero))

                    # Also check for near-zero values at sample points
                    near_zero_idx = np.where(np.abs(y_vals) < 1e-10)[0]
                    for idx in near_zero_idx:
                        x_val = sp.Float(valid_x[idx])
                        # Verify it's actually a zero by checking nearby points
                        # (avoid singularities that happen to evaluate near zero)
                        try:
                            y_val = float(f_lamb(float(x_val)))
                            eps = 1e-6
                            y_left = float(f_lamb(float(x_val) - eps))
                            y_right = float(f_lamb(float(x_val) + eps))
                            # Only add if function is continuous and actually crosses zero
                            if (
                                np.isfinite(y_left)
                                and np.isfinite(y_right)
                                and abs(y_val) < 1e-8
                            ):
                                # Avoid duplicates
                                if not any(
                                    abs(float(z - x_val)) < 1e-8
                                    for z in numerical_zeros
                                ):
                                    numerical_zeros.append(x_val)
                        except:
                            pass

                # Merge symbolic and numerical zeros, removing duplicates and singularities
                all_zeros = list(zeros)  # Start with symbolic zeros
                for nz in numerical_zeros:
                    # Check if this is actually a singularity
                    is_singularity = False
                    for sing in singularities:
                        try:
                            if abs(float(sing.evalf() - nz.evalf())) < 1e-6:
                                is_singularity = True
                                break
                        except:
                            pass

                    if is_singularity:
                        continue  # Skip singularities

                    # Check if this zero is already in the list (symbolic or numeric)
                    is_duplicate = False
                    for z in all_zeros:
                        try:
                            if abs(float(z.evalf() - nz.evalf())) < 1e-8:
                                is_duplicate = True
                                break
                        except:
                            pass
                    if not is_duplicate:
                        all_zeros.append(nz)

                zeros = all_zeros
        except:
            pass

    return {"zeros": zeros, "singularities": singularities}


def get_factors(polynomial, x):
    """Get factors for polynomial functions (legacy support)."""
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
        roots = sp.solve(linear_factor, x)

        # Handle factors that may have multiple real roots (e.g., quadratics like x^2 - 2)
        if not roots:
            linear_factors.append(
                {
                    "expression": linear_factor,
                    "exponent": exponent,
                    "root": -np.inf,
                }
            )
        else:
            # For each root of the factor, create a separate entry with the correct exponent
            for root_value in roots:
                # Only include real roots
                if root_value.is_real:
                    linear_factors.append(
                        {
                            "expression": sp.simplify(x - root_value),
                            "exponent": exponent,  # Use the actual exponent from factorization
                            "root": root_value,
                        }
                    )

    return linear_factors


def get_transcendental_factors(f, x, zeros, singularities):
    """Extract factors from transcendental or composite functions.

    Args:
        f: sympy expression
        x: variable symbol
        zeros: list of zeros found numerically/symbolically
        singularities: list of singularities (poles)

    Returns:
        list of factor dictionaries (one per unique expression+exponent, with all roots)
    """
    factors = []

    try:
        # Try to factor the function
        factored = sp.factor(f)

        # Check if it's a power expression (e.g., cos(x)**3)
        if factored.is_Pow:
            base = factored.base
            exponent = int(factored.exp) if factored.exp.is_integer else 1

            # Find zeros of the base
            base_zeros = []
            try:
                base_zeros_sym = sp.solve(base, x, domain=sp.S.Reals)
                for z in base_zeros_sym:
                    if z.is_real:
                        # Match with known zeros
                        for known_zero in zeros:
                            try:
                                if (
                                    abs(float(z.evalf()) - float(known_zero.evalf()))
                                    < 1e-8
                                ):
                                    base_zeros.append(known_zero)
                                    break
                            except:
                                pass
            except:
                # If symbolic solve fails, use all known zeros
                base_zeros = zeros

            # Create single factor with all zeros
            if base_zeros:
                factors.append(
                    {
                        "expression": base,
                        "exponent": exponent,
                        "roots": base_zeros,  # Changed: store all roots together
                    }
                )

        # If it's a product, try to extract individual factors
        elif factored.is_Mul:
            # Group factors by (expression, exponent) to avoid duplicates
            factor_dict = {}

            for arg in factored.args:
                # Check if this arg is a power
                if arg.is_Pow:
                    base = arg.base
                    exponent = int(arg.exp) if arg.exp.is_integer else 1
                    expr_to_use = base
                else:
                    expr_to_use = arg
                    exponent = 1

                # Create a key for grouping
                key = (str(expr_to_use), exponent)

                if key not in factor_dict:
                    factor_dict[key] = {
                        "expression": expr_to_use,
                        "exponent": exponent,
                        "zeros": [],
                        "singularities": [],
                    }

                # Find zeros of this factor
                try:
                    arg_zeros = sp.solve(expr_to_use, x, domain=sp.S.Reals)
                    for z in arg_zeros:
                        if z.is_real:
                            # Check if this zero is in our list
                            for known_zero in zeros:
                                try:
                                    if (
                                        abs(
                                            float(z.evalf()) - float(known_zero.evalf())
                                        )
                                        < 1e-8
                                    ):
                                        if known_zero not in factor_dict[key]["zeros"]:
                                            factor_dict[key]["zeros"].append(known_zero)
                                        break
                                except:
                                    pass
                except:
                    pass

                # Check for singularities in denominator
                try:
                    numer, denom = expr_to_use.as_numer_denom()
                    if denom != 1:
                        denom_zeros = sp.solve(denom, x, domain=sp.S.Reals)
                        for z in denom_zeros:
                            if z.is_real:
                                for known_sing in singularities:
                                    try:
                                        if (
                                            abs(
                                                float(z.evalf())
                                                - float(known_sing.evalf())
                                            )
                                            < 1e-8
                                        ):
                                            if (
                                                known_sing
                                                not in factor_dict[key]["singularities"]
                                            ):
                                                factor_dict[key][
                                                    "singularities"
                                                ].append(known_sing)
                                            break
                                    except:
                                        pass
                except:
                    pass

            # Convert grouped factors to list, with roots (zeros + singularities)
            for factor_info in factor_dict.values():
                all_roots = factor_info["zeros"] + factor_info["singularities"]
                if all_roots:
                    factors.append(
                        {
                            "expression": factor_info["expression"],
                            "exponent": factor_info["exponent"],
                            "roots": all_roots,
                        }
                    )
        else:
            # Not a product or power, show as single factor with all zeros
            all_roots = zeros + singularities
            if all_roots:
                factors.append(
                    {
                        "expression": factored,
                        "exponent": 1,
                        "roots": all_roots,
                    }
                )
    except:
        # Fallback: just use the original function
        all_roots = zeros + singularities
        if all_roots:
            factors.append(
                {
                    "expression": f,
                    "exponent": 1,
                    "roots": all_roots,
                }
            )

    return factors


def sort_factors(factors):
    def get_numeric_root(factor):
        # Handle both old format ("root") and new format ("roots")
        if "roots" in factor and factor["roots"]:
            # For new format, use the first root for sorting
            root = factor["roots"][0]
        else:
            root = factor.get("root")

        if root == -np.inf or root is None:
            return -np.inf
        try:
            # Try to convert symbolic roots to float for comparison
            return float(root.evalf())
        except (AttributeError, TypeError):
            try:
                return float(root)
            except:
                return -np.inf

    factors = sorted(factors, key=get_numeric_root)
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
        expression = factor.get("expression")
        exponent = factor.get("exponent")

        # Handle both old format (single "root") and new format (multiple "roots")
        factor_roots = (
            factor.get("roots", [factor.get("root")])
            if factor.get("root") is not None or factor.get("roots")
            else []
        )

        # Use LaTeX rendering for proper mathematical notation
        try:
            expression_latex = sp.latex(expression)
        except:
            # Fallback to string representation if latex fails
            expression_latex = str(expression)

        if exponent > 1:
            s = f"$({expression_latex})^{{{exponent}}}$"
        else:
            s = f"${expression_latex}$"

        plt.text(
            x=-0.1,
            y=(i + 1) * dy,
            s=s,
            fontsize=16,
            ha="right",
            va="center",
        )

        # If no real roots (constant factor)
        if -np.inf in factor_roots or not factor_roots:
            y_value = sp.sympify(expression).evalf(subs={x: 0})
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
        else:
            # Sort roots for drawing
            sorted_roots = sorted(
                [r for r in factor_roots if r != -np.inf],
                key=lambda r: float(r.evalf()),
            )

            # Determine if even exponent (doesn't change sign) or odd (changes sign)
            is_even_exponent = exponent % 2 == 0

            # Create the full expression with exponent for evaluation
            expr = expression
            if exponent > 1:
                full_expr = expr**exponent
            else:
                full_expr = expr

            # Draw segments between roots
            # We need to evaluate signs in each interval
            all_positions = (
                [x_min] + [root_positions[r] for r in sorted_roots] + [x_max]
            )

            for j in range(len(all_positions) - 1):
                left_pos = all_positions[j]
                right_pos = all_positions[j + 1]

                # Add gap around roots (except at boundaries)
                if j > 0:  # Not the leftmost segment
                    left_pos += dx
                if j < len(all_positions) - 2:  # Not the rightmost segment
                    right_pos -= dx

                # Map position back to actual x value for evaluation
                # Position is normalized 0-1, map to actual domain
                if j < len(sorted_roots):
                    if j == 0:
                        # Before first root
                        test_x = float(sorted_roots[0].evalf()) - 1
                    else:
                        # Between roots
                        test_x = (
                            float(sorted_roots[j - 1].evalf())
                            + float(sorted_roots[j].evalf())
                        ) / 2
                else:
                    # After last root
                    test_x = float(sorted_roots[-1].evalf()) + 1

                # Evaluate sign at test point
                try:
                    y_val = sp.sympify(full_expr).evalf(subs={x: test_x})
                    is_positive = y_val > 0
                    color = color_pos if is_positive else color_neg
                    linestyle = "-" if is_positive else "--"
                except:
                    # Fallback
                    color = color_pos
                    linestyle = "-"

                # Draw segment
                ax.plot(
                    [left_pos, right_pos],
                    [(i + 1) * dy, (i + 1) * dy],
                    color=color,
                    linestyle=linestyle,
                    lw=2,
                )

            # Draw markers at roots
            for root in sorted_roots:
                root_pos = root_positions[root]

                # Check if it's a zero or singularity
                try:
                    f_at_root = str(f.subs(x, root))
                    is_singularity = f_at_root == "zoo" or "zoo" in f_at_root
                except:
                    is_singularity = False

                if is_singularity:
                    plt.text(
                        x=root_pos + 0.005,
                        y=(i + 1) * dy,
                        s=f"$\\times$",
                        fontsize=24,
                        ha="center",
                        va="center",
                    )
                else:
                    plt.text(
                        x=root_pos,
                        y=(i + 1) * dy,
                        s=f"$0$",
                        fontsize=20,
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
            # Handle both old format ("root") and new format ("roots")
            factor_roots = (
                factor.get("roots", [factor.get("root")])
                if factor.get("root") is not None or factor.get("roots")
                else []
            )

            for root in factor_roots:
                if root != -np.inf:
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
    figsize=None,
    domain=None,
):
    """Draws a sign chart for a function f (polynomial, rational, or transcendental).

    Args:
        f (sp.Expr, str):
            Function expression. May be a sympy.Expr or str. Supports polynomials,
            rational functions, and transcendental functions (sin, cos, exp, log, etc.).
        x (sp.symbols, str, optional):
            Variable in the function
        fn_name (str, optional):
            Name of the function. Defaults `None`.
        color (bool, optional):
            Enables coloring of sign chart. Default: `True`.
        include_factors (bool, optional):
            Includes all linear factors of f(x) for polynomials. For non-polynomial
            functions, this shows the function name only. Default: `True`.
        generic_label (bool, optional):
            Uses generic labels for roots: x_1, x_2, ..., x_N. Default: `False`.
        small_figsize (bool, optional):
            Enables rescaling of the figure for a smaller figure size. Default: `False`.
        domain (tuple, optional):
            Domain (x_min, x_max) for searching zeros numerically in transcendental functions.
            If None, uses a default range or symbolic solving only. Example: (-10, 10)

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

    # Determine function type and extract zeros/singularities
    is_polynomial = f.is_polynomial()
    is_rational = False

    if is_polynomial:
        # Use existing polynomial factorization
        factors = get_factors(polynomial=f, x=x)
        factors = sort_factors(factors)
    else:
        # Check if it's a rational function
        try:
            numer, denom = f.as_numer_denom()
            if denom != 1 and denom.is_polynomial() and numer.is_polynomial():
                is_rational = True
                # Handle as rational function
                p_factors = get_factors(polynomial=numer, x=x) if numer != 1 else []
                q_factors = get_factors(polynomial=denom, x=x) if denom != 1 else []
                factors = p_factors + q_factors
                factors = sort_factors(factors)
            else:
                # General transcendental or composite function
                is_rational = False
                # Use general zero finding
                result = get_zeros_and_singularities(f, x, domain=domain)
                zeros = result["zeros"]
                singularities = result["singularities"]

                # Extract factors from transcendental function
                factors = get_transcendental_factors(f, x, zeros, singularities)
                factors = sort_factors(factors)
                # Now we can show factors for transcendental functions too!
        except:
            # Fallback: general function
            result = get_zeros_and_singularities(f, x, domain=domain)
            zeros = result["zeros"]
            singularities = result["singularities"]

            # Extract factors
            factors = get_transcendental_factors(f, x, zeros, singularities)
            factors = sort_factors(factors)

    print(f"Creating sign chart for f(x) = {f} = {f.factor()}")

    # Create figure
    fig, ax = make_axis(x)

    # Extract roots - handle both old format (single "root") and new format (multiple "roots")
    roots = []
    for factor in factors:
        if "roots" in factor and factor["roots"]:
            # New format: multiple roots per factor
            for r in factor["roots"]:
                if r != -np.inf and r not in roots:
                    roots.append(r)
        elif "root" in factor and factor["root"] != -np.inf:
            # Old format: single root per factor
            if factor["root"] not in roots:
                roots.append(factor["root"])

    # Sort roots
    roots = sorted(roots, key=lambda r: float(r.evalf()))

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
            f"${sp.latex(root)}$" if not generic_labels else f"$x_{i + 1}$"
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
        if figsize:
            fig.set_size_inches(figsize)
        else:
            fig.set_size_inches(8, 2 + int(0.7 * len(factors)))

    elif small_figsize:
        fig.set_size_inches(4, 1.5)
    else:
        fig.set_size_inches(8, 2)

    draw_vertical_lines(roots, root_positions, factors, ax, include_factors)

    plt.tight_layout()

    return fig, ax
