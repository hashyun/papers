"""Utilities to turn model evaluation results into publication-ready tables and figures.

The helpers in this module standardise styling decisions so that numbers produced by the
l1/l2 + GPD workflow can be dropped into a manuscript with minimal manual editing.  The
functions accept the raw ``pandas.DataFrame`` objects that are already created in the
notebook (``results_df``, ``full_df``, ``gof_df`` etc.) and return either formatted
``Styler`` instances or save Matplotlib figures.

Example
-------
>>> from publication_outputs import (
...     apply_publication_style, create_model_comparison_table,
...     style_table, plot_information_criteria, plot_risk_measures
... )
>>> apply_publication_style()
>>> model_table = create_model_comparison_table(full_df)
>>> styler = style_table(model_table, highlight_metric="AIC", precision={"AIC":1, "BIC":1})
>>> styler.to_latex("model_table.tex", hrules=True)
>>> plot_information_criteria(full_df, output_path="figures/information_criteria.png")
>>> plot_risk_measures(full_df, output_path="figures/risk_measures.png")
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

__all__ = [
    "apply_publication_style",
    "create_model_comparison_table",
    "style_table",
    "export_styler",
    "plot_information_criteria",
    "plot_risk_measures",
    "plot_tail_gof",
]

# ---------------------------------------------------------------------------
# Plot styling helpers
# ---------------------------------------------------------------------------
DEFAULT_FIGSIZE: Tuple[float, float] = (6.0, 4.0)
DEFAULT_PALETTE: Sequence[str] = (
    "#1f78b4",  # blue
    "#33a02c",  # green
    "#e31a1c",  # red
    "#ff7f00",  # orange
    "#6a3d9a",  # purple
    "#b15928",  # brown
)


def apply_publication_style(
    font: str = "DejaVu Serif",
    *,
    context: str = "paper",
    grid: bool = True,
    rc_overrides: Optional[Mapping[str, float]] = None,
) -> None:
    """Apply a consistent Matplotlib/Seaborn style for paper figures.

    Parameters
    ----------
    font:
        Base font family to use.  ``DejaVu Serif`` ships with Matplotlib and plays
        nicely with LaTeX exports.
    context:
        Seaborn context.  The default ``"paper"`` keeps labels compact.
    grid:
        Whether to show a subtle grid.  When ``False`` the background is plain white.
    rc_overrides:
        Optional dictionary with additional Matplotlib rc parameters.  This makes it
        easy to change figure sizes or fonts globally from the notebook.
    """

    sns.set_theme(
        context=context,
        style="whitegrid" if grid else "white",
        font=font,
        palette=DEFAULT_PALETTE,
        rc={
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "grid.color": "#cccccc",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            **(rc_overrides or {}),
        },
    )


# ---------------------------------------------------------------------------
# Table creation utilities
# ---------------------------------------------------------------------------
def _coerce_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with ``columns`` converted to numeric if possible."""

    coerced = df.copy()
    for col in columns:
        if col in coerced.columns:
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
    return coerced


def create_model_comparison_table(
    df: pd.DataFrame,
    *,
    sort_by: str = "AIC",
    precision: Optional[Mapping[str, int]] = None,
    include_rank: bool = True,
) -> pd.DataFrame:
    """Create a cleaned model comparison table sorted by ``sort_by``.

    Parameters
    ----------
    df:
        DataFrame returned by the notebook (``full_df``).  Must contain at least the
        columns ``Distribution`` and ``AIC``.
    sort_by:
        Column name used to sort the models.  Lower is considered better.
    precision:
        Optional mapping from column name to the number of decimals used for rounding.
    include_rank:
        When ``True`` a ``Rank`` column is added (1 = best according to ``sort_by``).

    Returns
    -------
    pandas.DataFrame
        A cleaned dataframe ready for styling/export.
    """

    numeric_cols = [c for c in ["Log-Likelihood", "AIC", "BIC", "VaR 99%", "ES 99%", sort_by] if c in df.columns]
    cleaned = _coerce_numeric_columns(df, numeric_cols)
    cleaned = cleaned.replace({np.inf: np.nan, -np.inf: np.nan})

    if sort_by in cleaned.columns:
        cleaned = cleaned.sort_values(sort_by, key=lambda s: s.fillna(np.inf)).reset_index(drop=True)

    if include_rank and sort_by in cleaned.columns:
        ranks = cleaned[sort_by].rank(method="min")
        cleaned.insert(0, "Rank", ranks.astype("Int64"))

    if precision:
        for col, prec in precision.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].round(prec)

    # Reorder columns for readability if available
    preferred_order = ["Rank", "Distribution", "Log-Likelihood", "AIC", "BIC", "VaR 99%", "ES 99%", "k"]
    ordered_cols = [c for c in preferred_order if c in cleaned.columns]
    remaining = [c for c in cleaned.columns if c not in ordered_cols]
    cleaned = cleaned[ordered_cols + remaining]

    return cleaned


def style_table(
    df: pd.DataFrame,
    *,
    highlight_metric: Optional[str] = None,
    precision: Optional[Mapping[str, int]] = None,
    thousands: Optional[str] = ",",
    caption: Optional[str] = None,
) -> pd.io.formats.style.Styler:
    """Create a ``Styler`` with uniform formatting and optional highlighting.

    Parameters
    ----------
    df:
        Table produced by :func:`create_model_comparison_table` or any compatible
        DataFrame.
    highlight_metric:
        Column for which the minimum value should be highlighted.  ``None`` disables
        highlighting.
    precision:
        Optional mapping from column name to number of decimals.  Non-specified
        columns default to ``2`` decimals for floats and no formatting for integers.
    thousands:
        Character used as thousands separator.  Pass ``None`` to disable.
    caption:
        Optional table caption.  Appears both in HTML and LaTeX exports.
    """

    styler = df.style

    fmt: MutableMapping[str, str] = {}
    if precision:
        for col, prec in precision.items():
            if col in df.columns:
                fmt[col] = f"{{:,.{prec}f}}" if thousands else f"{{:.{prec}f}}"

    # Apply default formatting to numeric columns not covered above
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in fmt:
            fmt[col] = "{:,}" if thousands else "{}"

    styler = styler.format(fmt)

    if highlight_metric and highlight_metric in df.columns:
        styler = styler.highlight_min(subset=[highlight_metric], color="#f4f3d5")

    table_styles = [
        {
            "selector": "th",
            "props": "font-weight: bold; background-color: #f5f5f5; border-bottom: 1px solid #bfbfbf;",
        },
        {
            "selector": "td",
            "props": "padding: 6px 12px; border-bottom: 1px solid #dddddd;",
        },
        {
            "selector": "caption",
            "props": "caption-side: top; font-weight: bold; font-size: 12pt;",
        },
    ]
    styler = styler.set_table_styles(table_styles)

    if caption:
        styler = styler.set_caption(caption)

    return styler


def export_styler(
    styler: pd.io.formats.style.Styler,
    path: Union[Path, str],
    *,
    latex: bool = False,
    **kwargs,
) -> Path:
    """Export a ``Styler`` to HTML or LaTeX depending on the extension.

    Parameters
    ----------
    styler:
        The styled table to export.
    path:
        Output file path.  Parent directories are created on demand.
    latex:
        Force LaTeX export regardless of extension.
    kwargs:
        Extra keyword arguments forwarded to :meth:`Styler.to_html` or
        :meth:`Styler.to_latex`.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if latex or path.suffix.lower() in {".tex", ".ltx"}:
        content = styler.to_latex(hrules=True, **kwargs)
    else:
        content = styler.to_html(**kwargs)

    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def _setup_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.margins(x=0.05)


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def _order_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric in df.columns:
        return df.sort_values(metric, key=lambda s: s.fillna(np.inf))
    return df


def plot_information_criteria(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("AIC", "BIC"),
    output_path: Optional[Union[Path, str]] = None,
    annotate: bool = True,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Plot bar charts for information criteria such as AIC/BIC."""

    fig, axes = plt.subplots(1, len(metrics), figsize=(figsize[0] * len(metrics), figsize[1]))
    if len(metrics) == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, metric in zip(axes, metrics):
        ordered = _order_by_metric(df, metric)
        sns.barplot(
            data=ordered,
            x="Distribution",
            y=metric,
            ax=ax,
            palette=DEFAULT_PALETTE,
        )
        _setup_axis(ax, f"{metric}", metric)
        if annotate:
            _annotate_bars(ax, fmt="{:.1f}")

    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

    return fig


def plot_risk_measures(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] = ("VaR 99%", "ES 99%"),
    output_path: Optional[Union[Path, str]] = None,
    annotate: bool = True,
    log_scale: bool = True,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
) -> plt.Figure:
    """Plot VaR/ES comparisons on an optional logarithmic scale."""

    fig, axes = plt.subplots(1, len(metrics), figsize=(figsize[0] * len(metrics), figsize[1]))
    if len(metrics) == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, metric in zip(axes, metrics):
        ordered = _order_by_metric(df, metric)
        sns.barplot(
            data=ordered,
            x="Distribution",
            y=metric,
            ax=ax,
            palette=DEFAULT_PALETTE,
        )
        _setup_axis(ax, metric, metric)
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}"))
        if annotate:
            _annotate_bars(ax, fmt="{:.0f}")

    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

    return fig


def plot_tail_gof(
    gof_df: pd.DataFrame,
    *,
    statistic_columns: Sequence[str] = ("KS Statistic", "AD Statistic"),
    output_path: Optional[Union[Path, str]] = None,
    figsize: Tuple[float, float] = (6.0, 3.5),
) -> plt.Figure:
    """Visualise leaf-level goodness-of-fit diagnostics.

    ``gof_df`` is the output of ``evaluation.gpd_gof_leafwise``.  Columns are expected
    to contain per-leaf KS/AD statistics plus metadata such as the number of excesses.
    """

    melted = gof_df.melt(id_vars=[c for c in gof_df.columns if c not in statistic_columns],
                         value_vars=list(statistic_columns),
                         var_name="Statistic",
                         value_name="Value")

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=melted,
        x="Leaf",
        y="Value",
        hue="Statistic",
        palette=DEFAULT_PALETTE[: len(statistic_columns)],
        ax=ax,
    )
    ax.set_xlabel("Leaf index")
    ax.set_ylabel("Test statistic")
    ax.set_title("Tail goodness-of-fit diagnostics")
    ax.legend(title="Statistic", frameon=False)
    ax.margins(x=0.05)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")

    return fig
