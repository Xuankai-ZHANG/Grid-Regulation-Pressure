# Grid Regulation Pressure Analysis

This repository contains code for analyzing grid regulation pressure in urban areas, including quantile regression modeling and data visualization components.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup](#setup)
- [Dependencies](#dependencies)
- [Reproduction Steps](#reproduction-steps)
- [Data Availability](#data-availability)
- [Code Availability](#code-availability)

## Project Overview
This project analyzes grid regulation pressure using quantile regression to estimate peak concentration (α) and ramp contribution (β) weights across different pressure scenarios. The code includes data processing, model fitting, and visualization components.

## Setup
1. Clone this repository to your local machine
2. Ensure Python 3.7+ is installed
3. Install the required dependencies (see below)
4. Place the dataset file `grid_data.xlsx` in the `data/` directory

## Dependencies
The project requires the following Python packages with recommended versions:
- numpy==2.2.6
- pandas==2.3.3
- statsmodels==0.14.5
- scikit-learn==1.7.2
- matplotlib==3.10.8 (for visualization)
- geopandas==0.15.0 (for mapping)
- seaborn==0.13.2 (for visualization)

You can install these dependencies using pip:
```bash
pip install numpy pandas statsmodels scikit-learn matplotlib geopandas seaborn
```

## Reproduction Steps

### 1. Run Quantile Regression Model
```bash
python quantile_regression.py
```
This will:
- Load data from `data/grid_data.xlsx`
- Run quantile regression for different τ values
- Generate output files in the `output_model/` directory:
  - `qr_fitting_evaluation.csv`
  - `qr_weights_results.pkl`
  - `qr_regression_report.txt`

**Running time**: Approximately a few seconds for the regression model.

### 2. Generate Visualizations
```bash
python plot_partA.py  # Generates Figure A1-A2
python plot_partB.py  # Generates Figure B1-B5
python plot_maps.py   # Generates maps
```
These scripts will generate various figures and maps in the `output_plot/` directory.

**Running time**: Plotting scripts run in seconds, while map generation may take around 1 minute.

## Data Availability
The datasets used in this study include Tokyo metropolitan grid-level electricity load data with 30-minute resolution, population mobility data based on mobile GPS, urban spatial structure data, and socioeconomic statistics. Electricity load and population mobility data are provided in anonymized, aggregated form and are subject to data sharing and privacy regulations.

To support research reproducibility, processed and de-identified datasets (including grid-level regulation pressure indicators and all data needed to reproduce the main research results) will be released in a public repository after the paper is published. For academic research purposes, additional data can be obtained through reasonable application with the approval of the data provider.

## Code Availability
All code for data processing, pressure decomposition, quantile regression modeling, and chart generation will be publicly released in this code repository after the paper is published.

## Directory Structure
```
Grid Regulation Pressure/
├── data/
│   └── grid_data.xlsx          # Input dataset
├── output/                     # Visualization outputs
├── output_model/               # Model output files
├── utils/
│   └── map_utils.py            # Utility functions for mapping
├── quantile_regression.py      # Main regression model
├── plot_partA.py               # Figure A1-A2 generation
├── plot_partB.py               # Figure B1-B5 generation
├── plot_maps.py                # Map generation
├── README.md                   # This file
└── LICENSE                     # License file
```

