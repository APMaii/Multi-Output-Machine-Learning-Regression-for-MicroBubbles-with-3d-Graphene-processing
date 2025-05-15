# Multi-Output Machine Learning Regression for MicroBubbles with 3D Graphene Processing

This project applies **multi-output machine learning regression** to analyze the behavior of **microbubbles** processed with **3D graphene structures**. The goal is to predict multiple physical properties of microbubbles under various experimental and fabrication conditions using traditional regression models.

---

## 📁 Project Structure

### `datasets/`
This directory contains three separate datasets:
- **`dataset1`**: Related to **pressure stability** of microbubbles.  
- **`dataset2`**: Focused on **drainage behavior** under various conditions.  
- **`dataset3`**: Contains data on **diameter** and **wall thickness** of microbubbles.  

Each dataset includes experimental features and their corresponding output parameters.

---

## 🧠 Main Components

- **`Main.py`**  
  The main pipeline for running multi-output regression models. This includes:
  - Loading datasets  
  - Preprocessing  
  - Model training  
  - Evaluation with regression metrics


- **`Hyperparameter_Values.py`**
  All hyperparameter values related to training stage for all models which are used in Main.py

- **`Plotting_codes1.py`**, **`Plotting_codes2.py`**, **`Plotting_codes3.py`**  
  These scripts generate visualizations such as:
  - R² scores and RMSE plots  
  - Predicted vs. actual values  
  - Performance comparison charts

- **`Test1.py` to `Test5.py`**  
  Initial configuration and backup scripts for early testing setups. These files contain exploratory tests and validations.

---

## 📊 Techniques Used

- Multi-output regression (e.g., MultiOutputRegressor)
- Traditional ML algorithms (e.g., Linear Regression, SVR, Random Forest)
- Metric evaluation (R², MAE, RMSE)
- Matplotlib and Seaborn for visualization

---
