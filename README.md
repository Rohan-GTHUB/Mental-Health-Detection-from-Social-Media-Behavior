# Mental-Health-Detection-from-Social-Media-Behavior
This machine learning project aims to predict whether a person is experiencing symptoms of depression based on their responses to behavioral and psychological questions related to social media usage.

# Problem Statement

Social media has become an integral part of daily life, but it also plays a role in influencing mental well-being. The goal of this project is to use survey-based data to predict depression levels using machine learning techniques — helping raise awareness and potentially flag mental health concerns early.

# Features Used

The dataset includes behavioral indicators like:

- Worry and anxiety levels
- Fluctuation in daily interests
- Social comparison through social media
- Difficulty concentrating
- Sleep issues and distraction levels
- Validation-seeking behavior
- Restlessness without social media
- Age and general mood patterns

# Target Variable

- `depressed` (binary):  
  - `1` → High risk of depression  
  - `0` → Low or no risk of depression

# Tools & Technologies

- **Python**, **pandas**, **NumPy**
- **Scikit-learn** (Random Forest, StandardScaler, metrics)
- **Seaborn** and **Matplotlib** for visualizations
- **Flask** for web deployment
- (Optional: **Streamlit** for interactive front-end)

##  Model Training

- Outliers removed using IQR
- Features scaled using `StandardScaler`
- Model: `RandomForestClassifier` with class balancing
- Achieved high accuracy with proper preprocessing and feature selection

## Deployment

A Flask-based web app allows users to input responses and receive real-time predictions on depression risk.



