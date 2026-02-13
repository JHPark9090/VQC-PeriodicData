"""
Data generators for VQC periodic advantage experiments.

Available generators:
    - narma_generator: NARMA time-series data
    - multisine_generator: Multi-sine (K sinusoids) data
    - mackey_glass_generator: Mackey-Glass chaotic time-series
    - adding_problem_generator: Adding problem (long-range dependency)
    - real_world_datasets: ETTh1, Weather, ECL forecasting benchmarks

Usage:
    from data.narma_generator import get_narma_data, get_narma_dataloaders
    from data.multisine_generator import get_multisine_data, get_multisine_dataloaders
    from data.mackey_glass_generator import get_mackey_glass_data, get_mackey_glass_dataloaders
    from data.adding_problem_generator import get_adding_data, get_adding_dataloaders
    from data.real_world_datasets import get_forecasting_dataloaders, get_etth1_dataloaders
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

from .multisine_generator import (
    generate_multisine_series,
    get_multisine_data,
    get_multisine_dataloaders,
    analyze_multisine_spectrum,
)

from .mackey_glass_generator import (
    generate_mackey_glass_series,
    get_mackey_glass_data,
    get_mackey_glass_dataloaders,
    analyze_mackey_glass_spectrum,
)

from .adding_problem_generator import (
    generate_adding_sequences,
    get_adding_data,
    get_adding_dataloaders,
)

from .real_world_datasets import (
    get_forecasting_dataloaders,
    get_etth1_dataloaders,
    get_weather_dataloaders,
    get_ecl_dataloaders,
    create_multivariate_sequences,
)

__all__ = [
    # NARMA
    'generate_narma_series',
    'generate_narma_variants',
    'create_narma_sequences',
    'get_narma_data',
    'get_narma_dataloaders',
    'get_narma_for_fourier_qlstm',
    'get_narma_for_fourier_qtcn',
    'analyze_narma_spectrum',
    'plot_narma_analysis',
    # Multi-Sine
    'generate_multisine_series',
    'get_multisine_data',
    'get_multisine_dataloaders',
    'analyze_multisine_spectrum',
    # Mackey-Glass
    'generate_mackey_glass_series',
    'get_mackey_glass_data',
    'get_mackey_glass_dataloaders',
    'analyze_mackey_glass_spectrum',
    # Adding Problem
    'generate_adding_sequences',
    'get_adding_data',
    'get_adding_dataloaders',
    # Real-World Forecasting
    'get_forecasting_dataloaders',
    'get_etth1_dataloaders',
    'get_weather_dataloaders',
    'get_ecl_dataloaders',
    'create_multivariate_sequences',
]
