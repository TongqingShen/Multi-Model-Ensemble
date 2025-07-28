# Multi-Model Ensemble for Climate Projection

This repository contains the official code of the article:

 **Enhancing Climate Projections via Machine Learning: Multi-Model Ensemble of Precipitation and Temperature in the Source Region of the Yellow River**  
 *Journal of Hydrology*, Vol. 662, 133945, 2025  
  
 **Authors**: Qi Ju, Jinyu Wu, Tongqing Shen, Yuying Wang, Huimin Cai, Jin Jin, Peng Jiang, Xinyu Chen, Yuxuan Du  
  
 **DOI**: [10.1016/j.jhydrol.2025.133945](https://doi.org/10.1016/j.jhydrol.2025.133945)

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

## Citation

If you use this code or data in your research, please cite:

 Ju, Q., Wu, J., Shen, T., Wang, Y., Cai, H., Jin, J., Jiang, P., Chen, X., Du, Y., 2025.  
 *Enhancing climate projections via machine learning: Multi-model ensemble of precipitation and temperature in the source region of the Yellow River*.  
 *Journal of Hydrology* 662, 133945.  
 [https://doi.org/10.1016/j.jhydrol.2025.133945](https://doi.org/10.1016/j.jhydrol.2025.133945)

---

## Contact

- **Corresponding author**: Jinyu Wu – wjy1231230916@163.com  
- **Repository maintained by**: [Tongqing Shen](https://github.com/TongqingShen)
