# Mini-PatchTST

A lightweight variant of the PatchTST Transformer architecture, designed for metro passenger demand forecasting with reduced model complexity.

## Overview

Mini-PatchTST applies patch-based time series modeling to metro ridership prediction. It uses a Transformer encoder over patched input sequences with configurable patch length, embedding dimension, and number of layers.

## Model Architecture

- **Patch embedding**: Input sequences are split into patches and projected to a latent dimension
- **Positional encoding**: Learnable positional embeddings
- **Transformer encoder**: Standard Transformer layers with multi-head attention
- **Forecast head**: Linear projection to the forecast horizon

Key hyperparameters: `patch_len`, `d_model`, `nhead`, `num_layers`, `input_length`, `forecast_horizon`.

## Installation

Requires Python ≥ 3.10. Using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Dependencies

- numpy
- pandas
- scikit-learn
- torch
- utilsforecast

## Usage

### Hangzhou Metro (Wei et al. version)

Runs configurations A_80 (epochs=110, num_layers=4) and B_5 (epochs=110, num_layers=2):

```bash
python forecast_hangzhou_dataset_version_Wei_et_al.py
```

### Hangzhou Metro (Hu version)

Per-station training for stations 9 and 15 (inbound/outbound):

```bash
python forecast_hangzhou_dataset_version_Hu.py
```

### Istanbul Metro

```bash
python forecast_istanbul_dataset.py
```

## Data

- **Hangzhou**: Raw metro records aggregated to 10-minute buckets by line, station, and direction (inbound/outbound)
- **Istanbul**: Hourly metro passenger flow data, with columns: `line_name`, `station_name`, `ds`, and `number_of_passengers`.

