# Regression and Neural Network Analysis

This repository contains a complete analysis of various regression techniques and a custom Artificial Neural Network (ANN) applied to both synthetic and real-world datasets. The project aims to understand the behavior of different models, evaluate their performance, and gain hands-on experience with data preprocessing, regularization, and neural network training.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Datasets](#datasets)  
3. [Models Implemented](#models-implemented)  
4. [Key Libraries](#key-libraries)  
5. [Jupyter Notebook](#jupyter-notebook)  
6. [Results and Visualizations](#results-and-visualizations)  
7. [Insights and Learnings](#insights-and-learnings)

---

## Project Overview

The project is divided into two main parts:

- **Synthetic Dataset Experiments:**  
  - Generated a cubic dataset: `y = 0.5x^3 - 2x^2 + 3x + 10 + noise`  
  - Compared Linear, Polynomial, Ridge, Lasso, ElasticNet regressions  
  - Built a custom ANN and experimented with different activation functions (tanh, LeakyReLU)  
  - Applied EarlyStopping to prevent overfitting  

- **California Housing Dataset Experiments:**  
  - Applied the same regression techniques and ANN on a real-world dataset  
  - Preprocessed features and targets using standardization and scaling  
  - Compared model performances using MSE and R² metrics  

---

## Datasets

1. **Synthetic Cubic Dataset:**  
   - 1 feature (X) with 1,000 samples  
   - Added Gaussian noise  

2. **California Housing Dataset:**  
   - 8 numerical features  
   - 20,640 samples (Train: 16,512, Test: 4,128)  
   - Target: median house value  

---

## Models Implemented

- Linear Regression  
- Polynomial Regression (degree=3)  
- Ridge Regression  
- Lasso Regression  
- ElasticNet  
- Custom Artificial Neural Network (ANN) with:
  - Input layer (depends on dataset features)  
  - Hidden layers: 128 → 64 neurons  
  - Activations: LeakyReLU (best performance)  
  - Output layer: 1 neuron (linear for synthetic, ReLU for California dataset)  
  - EarlyStopping to avoid overfitting  

---

## Key Libraries

- `numpy`, `pandas` – Data manipulation  
- `matplotlib` – Visualization  
- `scikit-learn` – Dataset loading, preprocessing, and traditional regression models  
- `tensorflow` / `keras` – Neural network implementation  
- `seaborn` (optional) – Enhanced plotting  

> These libraries are essential to reproduce the analysis.

---

## Jupyter Notebook

All code, plots, and analyses are implemented in a **Jupyter Notebook**:

- [Notebook Link (GitHub)](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)  

You can run the notebook cell by cell to reproduce the results and visualizations.

---

## Results and Visualizations

- Comparison of regression models on synthetic and real datasets (MSE, R²)  
- Coefficient analysis for Polynomial, Ridge, Lasso, and ElasticNet  
- ANN training curves and predicted vs actual plots  
- EarlyStopping behavior visualization  

Screenshots of plots and tables are included in the notebook.

---

## Insights and Learnings

- Polynomial features are essential to model nonlinear relationships.  
- Regularization improves stability but can affect performance depending on noise levels.  
- ANNs capture complex nonlinear patterns better than traditional regression models.  
- LeakyReLU showed better gradient flow and faster convergence than tanh.  
- EarlyStopping prevents overfitting and selects the best performing model automatically.  
- ANN performed best on the California Housing dataset, outperforming all linear models in terms of MSE and R².  

---

## License

This project is released under the MIT License.  

---

## Author

**Lyna** – [GitHub Profile](https://github.com/YOUR_USERNAME)
