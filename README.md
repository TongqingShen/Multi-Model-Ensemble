# Multi-Model Ensemble for Climate Projection

This repository contains the official code of the article:

> **Enhancing Climate Projections via Machine Learning: Multi-Model Ensemble of Precipitation and Temperature in the Source Region of the Yellow River**  
> *Journal of Hydrology*  
>  
> **Author**: Jinyu Wu 
> **Contact**: wjy1231230916@163.com  
> **Repository maintained by**: [Tongqing Shen](https://github.com/TongqingShen)

---

## 📘 Overview

This project provides a machine learning-based multi-model ensemble framework to improve climate projection accuracy. It focuses on precipitation and temperature prediction over the Source Region of the Yellow River (SRYR), utilizing outputs from multiple GCMs under CMIP6 scenarios.

The ensemble integrates three advanced regressors with Bayesian optimization:

- **Back Propagation Neural Network (BP)**
- **Long Short-Term Memory (LSTM)**
- **Random Forest (RF)**

Baseline ensemble via RMSE-weighted average is also implemented for comparison.

---

## 📁 Files

| File           | Description                             |
|----------------|-----------------------------------------|
| `multi_prec.py`| Ensemble modeling for **precipitation** |
| `multi_temp.py`| Ensemble modeling for **temperature**   |

Each script reads GCM data, performs normalization, trains ensemble models, applies Bayesian hyperparameter tuning, and generates future projections for SSP126, SSP245, and SSP585.

---
