# Li-ion Battery Performance Prediction

This repository contains a collection of codes for predicting the performance of Li-ion batteries based on charge and discharge cycle data. The dataset is obtained from an open-source website and includes cycle data from 14 different Li-ion batteries under constant voltage and constant current (CV-CC) conditions.

# Project Structure

**Dataset**: The dataset includes the charge and discharge cycle data for 14 Li-ion batteries, which is clubbed into a single dataset.

Dataset compiling credits: https://github.com/ignavinuales/Battery_RUL_Prediction
  
**Code Files**:
  1. `Feature_Extraction.py`: This script extracts statistical features like Standard Deviation (Std), Skewness, and Kurtosis from the given battery cycle data. 
     While it provides a new dataset, the results obtained from this feature extraction approach were not effective, and the data was not used further in the project.
  
  2. `FNN.py`: This is a feedforward neural network (FNN) implementation that uses manual hyperparameter tuning. The network consists of 6 layers and yields 
      impressive results, specifically:
      - **Mean Squared Error (MSE)**: 53.66
      - **Mean Absolute Error (MAE)**: 4.59
      - **R-squared (R²)**: 1.00
      - **Explained Variance Score**: 1.00
<p align="center">
  <img src="https://github.com/Buvaanesh/FNN_Battery_RUL/blob/main/Validation%20Loss%20-%20FNN.png" width="600" height="400">
</p>
    
  3. `Optuna_and_FNN.py`: This script is an enhanced version of the FNN model, which incorporates Optuna for hyperparameter optimization. The hyperparameter search yields improved results:
      - **Mean Squared Error (MSE)**: 71.93
      - **Mean Absolute Error (MAE)**: 4.64
      - **R-squared (R²)**: 1.00
      - **Explained Variance Score**: 1.00
<p align="center">
  <img src="https://github.com/Buvaanesh/FNN_Battery_RUL/blob/main/Validation%20Loss%20-%20Optuna%20and%20FNN.png"  width="600" height="400">
</p>

# Installation

To run this project locally, install these prerequisites:
- Python 3.x
- Libraries required:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tensorflow` (for neural network)
  - `optuna` (for hyperparameter optimization)
---
### **Contributing**

Contributions to improve the models or add new features are always welcome! If you'd like to contribute, please fork this repository, make your changes, and submit a pull request. For bug reports or feature requests, feel free to open an issue.

### Contact

If you have any questions, feel free to reach out to me directly or open an issue in this repository.

Thank you for checking out this project!

*Happy coding!* 🚀



