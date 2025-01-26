# Flood Susceptibility Mapping Project

## Overview
This project develops a machine learning model to predict flood susceptibility using geospatial data and Support Vector Machine (SVM) classification with Bayesian optimization.

## Project Description
Flood susceptibility mapping is a critical tool for understanding and mitigating flood risks. This research leverages multiple geographical factors to create a predictive model that can help identify areas prone to flooding.

## Key Features
- Multi-factor Geospatial Analysis
- Machine Learning Classification
- Bayesian Hyperparameter Optimization
- Feature Importance Evaluation

## Methodology

### Data Preprocessing
The project uses multiple geographical factors as input features:
- Distance from Road
- Drainage Density
- Distance from River
- Elevation
- Land Use/Land Cover (LULC)
- Normalized Difference Vegetation Index (NDVI)
- Slope
- Stream Power Index (SPI)
- Topographic Position Index (TPI)
- Topographic Wetness Index (TWI)

### Machine Learning Approach
- **Classifier**: Support Vector Machine (SVM)
- **Optimization**: Bayesian Search for Hyperparameter Tuning
- **Scaling**: MinMax Feature Scaling
- **Cross-Validation**: 5-fold Cross-Validation

## Prerequisites
- Python 3.8+
- Libraries:
  - NumPy
  - RasterIO
  - Scikit-learn
  - Scikit-Optimize

## Installation
```bash
pip install numpy rasterio scikit-learn scikit-optimize
