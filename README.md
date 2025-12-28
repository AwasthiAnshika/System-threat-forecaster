# System-threat-forecaster
A Machine Learning–based System Security Risk Prediction Model.
System Threat Forecaster is a Machine Learning project that predicts whether a computer system is at risk of a security threat based on system telemetry and configuration data.

The project uses a scalable linear classification approach to efficiently handle large, high-dimensional tabular data, making it suitable for real-world cybersecurity monitoring systems.This project demonstrates an end-to-end ML workflow including data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

Objectives:
1. Automatically analyze system-level data
2. Detect patterns indicating potential security threats
3. Predict whether a system is safe or at risk

Significance of using ML: Pattern recognition at scale, Automated and consistent threat detection, Improved system reliability and response time

Working of the project:
Loads and processes system telemetry data
Handles missing values and feature inconsistencies
Encodes and scales features for efficient learning
Trains a machine learning classifier
Optimizes model performance using cross-validation
Predicts system threat likelihood

Machine Learning Pipeline

Raw System Data (CSV)
↓
Data Cleaning and Imputation
↓
Feature Encoding and Scaling
↓
Train-Test Split
↓
SGD Classifier Training
↓
Hyperparameter Tuning using GridSearchCV
↓
Model Evaluation
↓
Threat Prediction

Features:
Mixed numerical and categorical system attributes, Target Variable: Binary classification- 0 indicates no threat and 1 indicates system at risk. The dataset is high-dimensional and requires efficient preprocessing and scalable models.

Data Preprocessing and Feature Engineering: Handling missing values using SimpleImputer; Encoding categorical variables for model compatibility; Feature scaling for stable gradient-based optimization; Ensuring clean and consistent input data for training

Models Used:SGDClassifier (Stochastic Gradient Descent); LightGBM (Gradient Boosting Machine); Naive Bayes

Hyperparameter optimization: Performed using GridSearchCV; Regularization techniques applied to improve generalization

Model Training and Evaluation: Train-test split ensures unbiased evaluation; Loss function: log loss or hinge loss

Optimizer: stochastic gradient descent

Evaluation metric: accuracy

Tech Stack: Programming Language: Python; Data Handling: Pandas, NumPy; Machine Learning: Scikit-learn; Visualization: Matplotlib; Environment: Jupyter Notebook, Kaggle

Results and Insights
1. Linear models can effectively detect system threats when properly tuned.
2. Data preprocessing plays a larger role than model complexity.
3. Scalable models are better suited for large telemetry datasets.

Future Improvements
1. Addition of ensemble models and techniques such as Random Forest or XGBoost.
2. Use advanced evaluation metrics like ROC-AUC and Precision-Recall.
3. Scalable to industry grade softwares.
