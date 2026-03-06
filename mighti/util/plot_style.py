"""
Central plotting defaults for MIGHTI.
"""

__all__ = ["apply_mighti_style"]


def apply_mighti_style(
    *,
    context="talk",
    font_scale=1.6,
    font_family="sans-serif",
    sans_serif=None,
    **overrides,
):
    """
    Apply a consistent Seaborn/Matplotlib style.

    Notes
    -----
    - Seaborn is optional; if unavailable we still set Matplotlib rcParams.
    - Safe/idempotent to call multiple times.
    """

    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_context(context, font_scale=font_scale)
    except Exception:
        pass

    if sans_serif is None:
        sans_serif = ["Arial", "DejaVu Sans", "Liberation Sans"]

    base = {
        "font.family": font_family,
        "font.sans-serif": sans_serif,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "axes.labelweight": "bold",
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "lines.linewidth": 2.5,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
    base.update(overrides)
    plt.rcParams.update(base)

