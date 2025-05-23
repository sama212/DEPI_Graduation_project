# Healthcare Predictive Analytics: Heart Disease Risk Prediction

![Heart Disease Prediction](https://www.labellerr.com/blog/content/images/2024/02/image--2-.webp)

## Overview

The Healthcare Predictive Analytics project is a comprehensive solution for predicting heart disease risk using machine learning techniques. Developed as part of the Digital Egypt Pioneers Initiative, this project aims to enhance healthcare outcomes through data-driven insights.

Cardiovascular diseases remain one of the leading causes of mortality worldwide, posing a significant burden on healthcare systems and affecting millions of lives annually. Early detection and effective risk assessment are essential to mitigating the impact of heart disease and improving patient outcomes.

This system serves as a decision-support tool for healthcare professionals, facilitating:
- Early detection of heart disease risk
- Prioritization of care based on risk levels
- Informed treatment planning
- Optimization of healthcare resource allocation

By analyzing patient attributes such as cholesterol levels, blood pressure, heart rate, and chest pain type, the project not only supports early intervention and prevention but also uncovers critical insights into the underlying risk factors contributing to cardiovascular conditions.

## Features

- **Risk Prediction**: Accurately predicts the likelihood of heart disease based on patient data
- **Feature Importance Analysis**: Identifies and visualizes the most significant factors contributing to heart disease risk
- **Interactive Dashboard**: User-friendly interface for inputting patient data and viewing predictions
- **Advanced Medical Assessment**: Provides comprehensive, medically sound explanations of prediction results
- **Data Visualization**: Graphical representation of patient data and risk factors
- **Cross-Validation**: Ensures model reliability and generalizability

## Dataset

The project utilizes the UCI Heart Disease dataset, which contains medical records related to heart disease diagnosis:

- **Size**: 920 rows, 16 columns
- **Source**: Kaggle (UCI Heart Disease Data)
- **Features**: Patient demographics, clinical measurements, and test results

### Key Attributes

| Feature | Description |
|---------|-------------|
| Age | Patient's age in years |
| Sex | Patient's gender (Male/Female) |
| CP | Chest pain type (Typical Angina, Atypical Angina, Non-Anginal, Asymptomatic) |
| Trestbps | Resting blood pressure (mm Hg) |
| Chol | Serum cholesterol (mg/dL) |
| FBS | Fasting blood sugar > 120 mg/dL (True/False) |
| RestECG | Resting electrocardiographic results (Normal, ST-T Abnormality, LV Hypertrophy) |
| Thalach | Maximum heart rate achieved |
| Exang | Exercise-induced angina (True/False) |
| Oldpeak | ST depression induced by exercise relative to rest |
| Slope | Slope of peak exercise ST segment |
| CA | Number of major vessels colored by fluoroscopy (0-3) |
| Thal | Thalassemia (Normal, Fixed Defect, Reversible Defect) |
| Num | Heart disease diagnosis (0-4) |

## Methodology

### 1. Data Preprocessing

- **Outlier Treatment**: Applied capping using defined valid ranges to handle extreme values without losing valuable data
- **Missing Values Handling**:
  - Predictive Imputation (Random Forest) for columns with high missing values
  - Mode Imputation for categorical columns with fewer missing values
  - Median Imputation for numerical columns with outliers
- **Categorical Variable Encoding**:
  - One-Hot Encoding for unordered categorical variables
  - Label Encoding for binary categorical variables
- **Feature Scaling**:
  - Normalization for non-normally distributed columns
  - Standardization for normally distributed columns with outliers

### 2. Exploratory Data Analysis

- Statistical analysis of all features
- Correlation analysis to identify relationships between variables
- Visualization of feature distributions and relationships
- Identification of key risk factors

### 3. Model Development

- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: Tested various configurations to optimize performance
- **Cross-Validation**: 5-fold cross-validation to ensure model reliability
- **Feature Importance Analysis**: Identified the most significant predictors of heart disease

### 4. Application Development

- Interactive Streamlit application for real-time predictions
- Visualization of prediction results and feature importance
- Advanced medical assessment and explanation of risk factors
- User-friendly interface for healthcare professionals

## Results and Insights

The model successfully identifies key risk factors for heart disease, including:

1. Number of major vessels colored by fluoroscopy (CA)
2. ST depression induced by exercise (Oldpeak)
3. Age
4. Maximum heart rate achieved (Thalach)
5. Chest pain type (CP)

These findings align with medical knowledge about cardiovascular disease risk factors and provide actionable insights for healthcare professionals.

## Installation and Usage

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository
2. Install required packages
3. Run the model training script
4. Launch the Streamlit application




## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Streamlit**: Web application development
- **Pickle**: Model serialization

## Contributors

This project was developed as part of the Digital Egypt Pioneers Initiative graduation project.

## Acknowledgments

- **Instructor**: Amgid Dife
- **Organized by**: Ministry of Communications and Information Technology
- **Implementing Company**: Eyouth
- **Initiative**: Digital Egypt Pioneers Initiative
