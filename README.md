# Papers Utilities

This repository collects helper scripts used when analysing Economic Policy
Uncertainty (EPU) data. In addition to the modelling code the project now
ships with utilities that streamline the creation of publication-ready
figures and tables.

## Publication helpers

The module `py/paper_formatting.py` provides a lightweight interface for
consistent styling across notebooks:

```python
from pathlib import Path

import pandas as pd

from paper_formatting import (
    apply_figure_style,
    dataframe_to_latex,
    prepare_table_dataframe,
    save_figure_for_publication,
    save_latex_table,
)

# 1. Configure matplotlib defaults once per notebook/script
apply_figure_style()

# 2. Generate a table ready for LaTeX
raw_results = pd.DataFrame({
    "Model": ["Baseline", "Proposed"],
    "Log-Likelihood": [-1234.56, -1198.31],
    "AIC": [2475.1, 2412.6],
})

table = prepare_table_dataframe(
    raw_results,
    column_order=["Model", "Log-Likelihood", "AIC"],
    float_format="{:.1f}",
)

latex = dataframe_to_latex(
    table,
    caption="Model comparison on EPU tail modelling.",
    label="tab:epu_models",
)
save_latex_table(latex, Path("outputs/tables/epu_models.tex"))

# 3. Save a matplotlib figure with tight layout and high DPI
# fig = ...
# save_figure_for_publication(fig, "outputs/figures/epu_tree.pdf")
```

These helpers avoid manual formatting in each notebook and guarantee that
exports share the same fonts, font sizes and resolution when inserted into a
paper or slide deck.
