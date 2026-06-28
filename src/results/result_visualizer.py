"""Compatibility layer for result visualizer under src/results.

This module re-exports visualization helpers from src/visualization so callers
can import either path without changing legacy code.
"""

from __future__ import annotations

from src.visualization.result_visualizer import (
    add_rank_column,
    format_results_table,
    generate_results_summary,
    load_results_csv,
    plot_accuracy_bar_chart,
    plot_method_average_rank_bar_chart,
    plot_rmse_bar_chart,
    run_result_visualization,
    save_formatted_table,
    save_wilcoxon_significance_table,
    sort_results_by_method_order,
    sort_results_by_rmse,
)

__all__ = [
    "load_results_csv",
    "sort_results_by_rmse",
    "add_rank_column",
    "format_results_table",
    "save_formatted_table",
    "plot_rmse_bar_chart",
    "plot_accuracy_bar_chart",
    "plot_method_average_rank_bar_chart",
    "save_wilcoxon_significance_table",
    "generate_results_summary",
    "sort_results_by_method_order",
    "run_result_visualization",
]
