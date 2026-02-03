"""
Data generators for VQC periodic advantage experiments.

Available generators:
    - narma_generator: NARMA time-series data

Usage:
    from data.narma_generator import get_narma_data, get_narma_dataloaders
"""

from .narma_generator import (
    generate_narma_series,
    generate_narma_variants,
    create_narma_sequences,
    get_narma_data,
    get_narma_dataloaders,
    get_narma_for_fourier_qlstm,
    get_narma_for_fourier_qtcn,
    analyze_narma_spectrum,
    plot_narma_analysis
)

__all__ = [
    'generate_narma_series',
    'generate_narma_variants',
    'create_narma_sequences',
    'get_narma_data',
    'get_narma_dataloaders',
    'get_narma_for_fourier_qlstm',
    'get_narma_for_fourier_qtcn',
    'analyze_narma_spectrum',
    'plot_narma_analysis'
]
