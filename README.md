# Battle Duration Prediction

![Python](https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20XGBoost-orange?style=flat-square)
![Data](https://img.shields.io/badge/Data-Polars-yellow?style=flat-square)
![Task](https://img.shields.io/badge/Task-Regression-success?style=flat-square)

A machine learning project aimed at predicting battle duration (`duration`) based on game replay data. The repository contains a pipeline for data processing, hyperparameter tuning, and training various model architectures.

## Data
* **Processing**: Optimized using the `Polars` library.
* **Filtering**: Data is mapped and split into subsets based on unique maps (`map_id`), allowing for the training of dedicated models for individual maps.

## Applied Models

The project experiments with two main approaches to tabular data:

### 1. Classical Machine Learning (Script 1)
* **XGBoost (XGBRegressor)**: A decision tree-based algorithm with Early Stopping.
* **SVR (Support Vector Regression)**: A support vector machine model utilizing feature scaling (StandardScaler).

### 2. Deep Learning (Script 2)
* **DCN + MLP (Deep & Cross Network)**: Implemented in PyTorch. The model uses an Embedding layer for map identifiers and then processes numerical and categorical features in parallel through:
  * **Cross Network**: For modeling feature interactions of various orders.
  * **Deep Neural Network (MLP)**: For learning non-linear patterns.
* **Loss Function**: `Cauchy Loss` – a loss function robust to outliers.

## Architecture Rationale: DCN vs XGBoost

The project implements a full **DCN (Deep & Cross Network)** architecture combined with an **MLP** (PyTorch file). The code defines a `CrossNetwork` and `nn.Sequential` (for the MLP), and the outputs of both networks are concatenated.

**Analysis of DCN applicability:**
* In the context of tabular data with a small number of categorical features (mainly `map_id` mapping alongside predominantly numerical features), the complex DCN architecture represents an advanced approach that may exceed the required model complexity.
* Decision tree-based models (like the XGBoost implemented in the repository) typically achieve higher effectiveness (`R²`) in these types of tasks with significantly shorter training times and less need for hyperparameter tuning.
* DCN architecture achieves the best results in recommendation systems with a large number of sparse categorical features. In this project, the DCN with MLP model serves as an advanced **benchmark**, allowing for a direct comparison of the effectiveness and computational cost between classical machine learning methods and deep neural networks.

## Optimization
**Optuna** was used for Hyperparameter Tuning in both scripts, aiming to minimize the Mean Absolute Error (MAE) and Cauchy Loss for PyTorch.

## Requirements and Setup

The project uses the `uv` package manager. Main dependencies include:
* `polars`
* `scikit-learn`
* `xgboost`
* `torch`
* `optuna`
* `tabulate`
