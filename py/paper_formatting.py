"""Utility helpers for producing publication-ready tables and figures.

This module provides two primary sets of functionality:

1.  Consistent Matplotlib styling so that plots generated across notebooks
    share the same font family, font sizes and DPI when exported.
2.  Helpers that turn :class:`pandas.DataFrame` objects into polished
    LaTeX tables with booktabs rules and optional highlighting of key
    rows/columns.

The goal is to minimise manual formatting work when adding outputs to a
paper.  Functions in this module avoid heavy optional dependencies and rely
solely on packages that already exist in the analysis environment
(`matplotlib` and `pandas`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Figure styling utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperFigureStyle:
    """Container describing default styling choices for plots.

    Parameters
    ----------
    font_family:
        Main font family used for titles and axis labels.
    base_font_size:
        Default font size applied to most textual elements.
    axis_label_size:
        Size of axis labels.  If ``None`` the value is derived from
        ``base_font_size``.
    tick_label_size:
        Tick label font size.  Defaults to slightly smaller than
        ``base_font_size``.
    legend_font_size:
        Font size for legend text.
    figure_size:
        Default figure size (width, height) in inches.  Individual plotting
        functions are free to override this.
    dpi:
        Resolution for rendered figures, used both for interactive sessions
        and when saving to disk.
    color_cycle:
        Optional iterable of colours to use for the axes colour cycle.  When
        ``None`` Matplotlib's default colour cycle is used.
    """

    font_family: str = "Times New Roman"
    base_font_size: float = 10.0
    axis_label_size: Optional[float] = None
    tick_label_size: Optional[float] = None
    legend_font_size: Optional[float] = None
    figure_size: tuple[float, float] = (6.0, 4.0)
    dpi: int = 300
    color_cycle: Optional[Sequence[str]] = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )


DEFAULT_FIGURE_STYLE = PaperFigureStyle()


def apply_figure_style(style: PaperFigureStyle = DEFAULT_FIGURE_STYLE) -> None:
    """Apply a consistent, publication-friendly Matplotlib style.

    The function updates :mod:`matplotlib`'s global rcParams so that any
    subsequent plots inherit the configuration.  It can be called once at the
    beginning of a notebook or script.

    Parameters
    ----------
    style:
        Optional :class:`PaperFigureStyle` instance describing the styling
        choices.  When omitted, :data:`DEFAULT_FIGURE_STYLE` is used.
    """

    axis_label_size = style.axis_label_size or style.base_font_size + 1
    tick_label_size = style.tick_label_size or style.base_font_size - 1
    legend_font_size = style.legend_font_size or style.base_font_size - 0.5

    mpl.rcParams.update(
        {
            "font.family": style.font_family,
            "font.size": style.base_font_size,
            "axes.titlesize": style.base_font_size + 1,
            "axes.labelsize": axis_label_size,
            "xtick.labelsize": tick_label_size,
            "ytick.labelsize": tick_label_size,
            "legend.fontsize": legend_font_size,
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
        }
    )

    if style.color_cycle is not None:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(style.color_cycle))


def save_figure_for_publication(
    figure: plt.Figure,
    path: Path | str,
    *,
    tight_layout: bool = True,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.05,
) -> None:
    """Persist a Matplotlib figure using recommended export settings.

    Parameters
    ----------
    figure:
        The figure object to save.
    path:
        Destination file path.  Parent directories are created automatically.
    tight_layout:
        When ``True`` ``figure.tight_layout()`` is invoked prior to saving to
        minimise unused whitespace.
    bbox_inches:
        Argument forwarded to :meth:`matplotlib.figure.Figure.savefig`.
    pad_inches:
        Controls the amount of padding around the figure when ``bbox_inches``
        is set to ``"tight"``.
    """

    if tight_layout:
        figure.tight_layout()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches=bbox_inches, pad_inches=pad_inches)


# ---------------------------------------------------------------------------
# Tabular formatting utilities
# ---------------------------------------------------------------------------

def prepare_table_dataframe(
    df: pd.DataFrame,
    *,
    column_order: Optional[Sequence[str]] = None,
    column_renames: Optional[Mapping[str, str]] = None,
    float_format: str = "{:.2f}",
    percentage_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return a copy of *df* with common paper-friendly formatting applied.

    The helper performs non-destructive operations such as reordering columns,
    renaming headers, and rendering floating point columns using a uniform
    format string.  Percentage columns (supplied via ``percentage_columns``)
    are multiplied by 100 and suffixed with ``%``.
    """

    formatted = df.copy()

    if column_renames:
        formatted = formatted.rename(columns=column_renames)

    if column_order:
        formatted = formatted.loc[:, list(column_order)]

    fmt_cols = formatted.select_dtypes(include=["float", "float64", "float32"]).columns

    def _format_value(value: float, as_percentage: bool) -> str:
        if pd.isna(value):
            return "--"
        if as_percentage:
            return float_format.format(value * 100) + "%"
        return float_format.format(value)

    for col in fmt_cols:
        as_percentage = bool(percentage_columns and col in percentage_columns)
        formatted[col] = formatted[col].map(lambda x, ap=as_percentage: _format_value(x, ap))

    return formatted


def dataframe_to_latex(
    df: pd.DataFrame,
    *,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    column_format: Optional[str] = None,
    index: bool = False,
    bold_rows: Optional[Iterable[int]] = None,
    bold_columns: Optional[Iterable[str]] = None,
    escape: bool = True,
) -> str:
    """Generate a LaTeX table string suitable for academic writing.

    Parameters
    ----------
    df:
        The table data.  The function operates on a copy to avoid modifying
        the caller's object.
    caption:
        LaTeX caption placed above the table when used with ``booktabs``.
    label:
        Optional label for cross-referencing (e.g. ``"tab:results"``).
    column_format:
        Custom ``\begin{tabular}{...}`` column specification.  When omitted
        a simple left/right alignment is inferred based on the DataFrame.
    index:
        Whether to include the index in the table output.
    bold_rows / bold_columns:
        Indices or column names whose values should be wrapped in ``\textbf``.
    escape:
        Controls whether special LaTeX characters are escaped.
    """

    table = df.copy()

    if bold_rows:
        for idx in bold_rows:
            if idx in table.index:
                table.loc[idx] = table.loc[idx].map(_wrap_bold)

    if bold_columns:
        for col in bold_columns:
            if col in table.columns:
                table[col] = table[col].map(_wrap_bold)

    latex = table.to_latex(
        index=index,
        caption=caption,
        label=label,
        escape=escape,
        column_format=column_format,
        bold_rows=False,
        longtable=False,
        multicolumn=True,
        multicolumn_format="c",
        sparsify=False,
        header=True,
        na_rep="--",
        float_format=None,
        buf=None,
    )

    return latex


def save_latex_table(
    latex_str: str,
    path: Path | str,
) -> None:
    """Write the given LaTeX table string to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_str, encoding="utf-8")


def _wrap_bold(value: object) -> object:
    """Wrap a value with ``\textbf{}`` if it is a string, otherwise return it."""

    text = str(value)
    return f"\\textbf{{{text}}}"


__all__ = [
    "PaperFigureStyle",
    "DEFAULT_FIGURE_STYLE",
    "apply_figure_style",
    "save_figure_for_publication",
    "prepare_table_dataframe",
    "dataframe_to_latex",
    "save_latex_table",
]

