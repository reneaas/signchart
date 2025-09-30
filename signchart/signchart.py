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
    fontsize=14,
    factor_fontscale=0.9,
    zero_fontscale=1.1,
    zero_x_offset=0.0,
    zero_y_offset=0.0,
    cross_x_offset=0.0,
    cross_y_offset=0.0,
):
    x_min = -0.05
    x_max = 1.05
    # Draw horizontal sign lines for each factor
    for i, factor in enumerate(factors):
        expression = str(factor.get("expression"))
        exponent = factor.get("exponent")

        # Replace ** with ^ for sympy expressions to work with latex
        if "**" in str(expression):
            expression = expression.replace("**", "^")

        # Remove multiplication signs
        if "*" in str(expression):
            expression = expression.replace("*", "")

        if exponent > 1:
            s = f"$({expression})^{exponent}$"
        else:
            s = f"${expression}$"

        plt.text(
            x=-0.1,
            y=(i + 1) * dy,
            s=s,
            fontsize=fontsize * factor_fontscale,
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
                    x=root_pos + zero_x_offset,
                    y=(i + 1) * dy,
                    s=f"$0$",
                    fontsize=fontsize * zero_fontscale,
                    ha="center",
                    va="center",
                )
            else:
                plt.text(
                    x=root_pos + cross_x_offset,
                    y=(i + 1) * dy,
                    s=f"$\\times$",
                    fontsize=fontsize * zero_fontscale * 1.2,
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

            plt.text(
                x=root_pos + zero_x_offset,
                y=(i + 1) * dy,
                s=f"$0$",
                fontsize=fontsize * zero_fontscale,
                ha="center",
                va="center",
            )

            # if str(f.subs(x, root)) != "zoo":
            #     plt.text(
            #         x=root_pos,
            #         y=(i + 1) * dy,
            #         s=f"$0$",
            #         fontsize=20,
            #         ha="center",
            #         va="center",
            #     )
            # else:
            #     plt.text(
            #         x=root_pos + 0.005,
            #         y=(i + 1) * dy,
            #         s=f"$\\times$",
            #         fontsize=24,
            #         ha="center",
            #         va="center",
            #     )


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
    fontsize=14,
    function_fontscale=0.95,
    zero_fontscale=1.1,
    zero_x_offset=0.0,
    cross_x_offset=0.0,
    cross_y_offset=0.0,
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
        fontsize=fontsize * function_fontscale,
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
                x=root_pos + zero_x_offset,
                y=y,
                s=f"$0$",
                fontsize=fontsize * zero_fontscale,
                ha="center",
                va="center",
            )
        else:
            plt.text(
                x=root_pos + cross_x_offset,
                y=y + cross_y_offset,
                s=f"$\\times$",
                fontsize=fontsize * zero_fontscale * 1.2,
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
    symbol_gap=0.15,
):
    """Draw vertical separator lines that connect the x-axis (y=0) to each zero/cross symbol.

    We compute all symbol y positions at the given root and draw individual segments between
    consecutive symbol rows (centered vertically between the horizontal sign lines) so that
    each vertical part ends exactly at the symbol center. This avoids awkward gaps and ensures
    alignment.
    """

    if include_factors:
        symbol_rows = [
            (i + 1) * dy
            for i, factor in enumerate(factors)
            if factor.get("root") != -np.inf
        ]
        function_row = (len(factors) + 1) * dy
    else:
        symbol_rows = []
        function_row = dy

    all_symbol_rows = sorted(symbol_rows + [function_row], reverse=True)
    if not all_symbol_rows:
        return

    for root in roots:
        root_pos = root_positions[root]
        prev_y = 0  # start at axis (y=0)
        for row in all_symbol_rows:
            upper = row + symbol_gap
            lower = row - symbol_gap
            # Draw from previous segment end to just above symbol
            if prev_y > upper:  # when dy is negative prev_y closer to 0 is larger value
                ax.plot([root_pos, root_pos], [prev_y, upper], color="black", lw=1)
            else:
                ax.plot([root_pos, root_pos], [upper, prev_y], color="black", lw=1)
            prev_y = lower
        # tail below last symbol for visual balance
        tail = abs(dy) * 0.15
        ax.plot([root_pos, root_pos], [prev_y, prev_y - tail], color="black", lw=1)


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
    ax.set_xlabel(f"${str(x)}$", fontsize=20, loc="right")

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
    figsize=(6, None),
    fontsize=20,
    line_height=2,
    auto_height=True,
    dpi=300,
    adjust_root_labels=True,
    symbol_gap_scale=None,
    zero_x_offset=0.0,
    cross_x_offset=0.0,
    cross_y_offset=0.04,
    dynamic_layout=True,
):
    """Draws a sign chart (sign diagram) for a polynomial or rational function f.

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
        figsize (tuple(float,float|None)): (width, height). If height is None and auto_height=True,
            a suitable height is computed from number of rows and fontsize.
        fontsize (int): Base fontsize for textual elements (points).
        line_height (float): Multiplier converting fontsize to per-row vertical spacing (in points).
        auto_height (bool): If True, compute / override figure height based on content.
        dpi (int): Figure DPI.
        adjust_root_labels (bool): Shrink root tick labels if they would overflow horizontally.
        symbol_gap_scale (float|None): If float, manual fraction of row spacing reserved as vertical gap around
            symbols for vertical lines (0 < value < 0.5). If None, an automatic value is derived from line_height.
        zero_x_offset (float): Horizontal fine adjustment for the '0' glyph (data units).
        cross_x_offset (float): Horizontal fine adjustment for the '×' symbol (data units).
        cross_y_offset (float): Vertical fine adjustment for '×' (data units).
        dynamic_layout (bool): If True, compute horizontal padding (dx) and symbol gap automatically from
            figure size & fontsize. Disabling reverts to legacy static spacing.

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

    if small_figsize:
        warnings.warn(
            "'small_figsize' argument is deprecated; please directly provide 'figsize'.",
            DeprecationWarning,
        )

    if not f.is_polynomial():
        # Assume a rational function
        p, q = str(f).split("/")
        p = sp.sympify(p)
        q = sp.sympify(q)

        try:
            original_variable = list(p.free_symbols)[0]
            p = p.subs(original_variable, x)
            # Order the factors
            p_factors = get_factors(polynomial=p, x=x)
        except:
            p_factors = []

        try:
            original_variable = list(q.free_symbols)[0]
            q = q.subs(original_variable, x)
            q_factors = get_factors(polynomial=q, x=x)
        except:
            q_factors = []

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
    # Determine tick label fontsize adaptively if requested
    tick_label_fontsize = fontsize
    if adjust_root_labels and len(roots) > 0 and figsize is not None:
        # Rough heuristic: total label width (chars * 0.6 * fontsize) should be < 0.85 * fig width in points
        fig_width_pts = figsize[0] * 72
        total_estimated_pts = 0
        for root in roots:
            label = f"{sp.latex(sp.sympify(root))}" if not generic_labels else "x_i"
            total_estimated_pts += len(label) * 0.6 * fontsize
        if total_estimated_pts > 0.85 * fig_width_pts:
            scale = 0.85 * fig_width_pts / total_estimated_pts
            tick_label_fontsize = max(6, fontsize * scale)

    plt.xticks(
        ticks=positions,
        labels=[
            f"${sp.latex(sp.sympify(root))}$" if not generic_labels else f"$x_{i + 1}$"
            for i, root in enumerate(roots)
        ],
        fontsize=tick_label_fontsize,
    )

    # --- Figure sizing strategy (pre-spacing calculations) ---
    if figsize is None:
        rows = (len(factors) + 1) if include_factors else 1
        row_height_in = (fontsize * line_height) / 72.0
        base_padding_in = 0.6
        auto_h = base_padding_in + rows * row_height_in
        fig_w = 8
        fig_h = auto_h if auto_height else 2 + 0.7 * len(factors)
        figsize = (fig_w, fig_h)
    elif auto_height and (figsize[1] is None or isinstance(figsize[1], type(None))):
        # Provided only width
        rows = (len(factors) + 1) if include_factors else 1
        row_height_in = (fontsize * line_height) / 72.0
        base_padding_in = 0.6
        fig_h = base_padding_in + rows * row_height_in
        figsize = (figsize[0], fig_h)

    fig.set_dpi(dpi)
    fig.set_size_inches(figsize)

    # Horizontal dynamic spacing around roots (dx)
    if len(roots) > 0:
        avg_chars = np.mean(
            [len(sp.latex(sp.sympify(r))) if not generic_labels else 3 for r in roots]
        )
        label_width_pts = max(1, avg_chars * 0.6 * tick_label_fontsize)
        label_width_in = label_width_pts / 72.0
        fig_w_in = figsize[0]
        x_range = 1.10  # data span from -0.05 to 1.05
        approx_label_data = (label_width_in / fig_w_in) * x_range
        dynamic_dx = np.clip(approx_label_data * 0.35, 0.007, 0.08)
    else:
        dynamic_dx = 0.02

    if not dynamic_layout:
        dynamic_dx = 0.02  # revert to legacy fixed spacing

    # Automatic symbol gap fraction if not provided
    if symbol_gap_scale is None:
        # A heuristic: smaller line_height => tighter rows => need smaller relative gap.
        # Gap fraction roughly half the inverse of line_height, clamped.
        auto_gap = 0.5 / max(1.0, line_height)
        symbol_gap_fraction = float(np.clip(auto_gap, 0.18, 0.42))
    else:
        symbol_gap_fraction = float(np.clip(symbol_gap_scale, 0.05, 0.49))

    # Minor automatic cross vertical fine-tune if dynamic_layout and user left default
    if dynamic_layout and cross_y_offset == 0.04:
        cross_y_offset = 0.06 / line_height

    # Draw factors
    if include_factors:
        draw_factors(
            f,
            factors,
            roots,
            root_positions,
            ax,
            color_pos,
            color_neg,
            x,
            fontsize=fontsize,
            zero_x_offset=zero_x_offset,
            cross_x_offset=cross_x_offset,
            dx=dynamic_dx,
        )

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
        fontsize=fontsize,
        zero_x_offset=zero_x_offset,
        cross_x_offset=cross_x_offset,
        cross_y_offset=cross_y_offset,
        dx=dynamic_dx,
    )

    # Remove tick labels on y-axis
    plt.yticks([])

    plt.xlim(x_min, x_max)

    # symbol_gap_fraction is a fraction of the row spacing (|dy| = 1). Since dy is -1, fraction == data units.
    symbol_gap = symbol_gap_fraction
    draw_vertical_lines(
        roots,
        root_positions,
        factors,
        ax,
        include_factors,
        dy=-1,
        symbol_gap=symbol_gap,
    )

    plt.tight_layout()

    return fig, ax
