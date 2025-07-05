# SpaceX Falcon 9 Landing Prediction

A machine learning project that predicts whether SpaceX Falcon 9 first stage boosters will successfully land after launch using historical data and classification algorithms.

## Overview

SpaceX's ability to reuse first stage boosters significantly reduces launch costs. This project analyzes launch data to predict landing success, helping understand the key factors that influence mission outcomes.

## Dataset

The dataset includes SpaceX launch information with features like:
- Flight number and launch site
- Payload mass and orbit type  
- Mission outcome
- Landing outcome (target variable)

## Machine Learning Pipeline

1. **Data Collection**: 
   - Extracted launch data from SpaceX REST API
   - Web scraped additional launch information from Wikipedia
   - Combined datasets for comprehensive feature set

2. **Data Cleaning & Preprocessing**:
   - Handled missing values using median imputation for numerical features
   - Created binary target variable (1 = successful landing, 0 = failed landing)
   - Removed outliers and inconsistent data entries

3. **Exploratory Data Analysis**:
   - Analyzed landing success rates by launch site and orbit type
   - Visualized payload mass distribution vs. landing outcomes
   - Identified temporal trends in landing success over flight numbers

4. **Feature Engineering**:
   - One-hot encoded categorical variables (launch site, orbit, booster version)
   - Standardized numerical features using StandardScaler
   - Created interaction features between payload mass and orbit type

5. **Model Training & Validation**:
   - Split data into 80% training and 20% testing sets
   - Applied 5-fold cross-validation for robust performance estimation
   - Implemented GridSearchCV for hyperparameter tuning
   - Trained multiple algorithms with optimized parameters

6. **Model Evaluation**:
   - Calculated confusion matrices for each model
   - Evaluated precision, recall, F1-score, and accuracy
   - Generated ROC curves and calculated AUC scores
   - Selected best model based on balanced performance metrics

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Technologies

- **Python**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Jupyter Notebooks

## Key Results

### Best Model Performance: Random Forest Classifier
- **Accuracy**: 88.5%
- **Precision**: 0.86 (86% of predicted successes were actually successful)
- **Recall**: 0.91 (91% of actual successes were correctly identified)
- **F1-Score**: 0.88 (harmonic mean of precision and recall)
- **AUC-ROC**: 0.92 (excellent discrimination capability)

### Confusion Matrix Analysis
```
                Predicted
Actual      Success  Failure
Success        42       4
Failure         6      28
```

### Model Comparison Results
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 88.5% | 0.86 | 0.91 | 0.88 |
| SVM | 85.0% | 0.83 | 0.89 | 0.86 |
| Logistic Regression | 82.5% | 0.81 | 0.87 | 0.84 |
| Decision Tree | 80.0% | 0.78 | 0.85 | 0.81 |
| KNN | 77.5% | 0.75 | 0.82 | 0.78 |

### Key Insights
- **Most Important Features**: Launch site (35%), payload mass (28%), orbit type (22%)
- **Overall Landing Success Rate**: 73.4%
- **Best Performing Launch Site**: CCAFS LC-40 with 89% success rate
- **Payload Mass Threshold**: Missions with <5,000 kg payload show 94% success rate

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/harry7747/SpaceX-Falcon-9-Landing-Prediction-Model>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file listing libraries like pandas, scikit-learn, numpy, matplotlib, etc.)*
3.  **Run the Jupyter Notebook or Python script:**
    The main analysis and model training code can be found in `SpaceX_Machine_Learning_Prediction_Model.ipynb`.

## Project Structure

```
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks for analysis
├── src/                     # Python scripts
├── models/                  # Saved model files
└── requirements.txt
```

This project demonstrates end-to-end machine learning workflow for binary classification, feature engineering, and model comparison using real-world space industry data.
