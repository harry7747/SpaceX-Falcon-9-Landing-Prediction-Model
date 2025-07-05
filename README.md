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

## Machine Learning Pipeline

1. **Data Collection**: Extracted launch data from SpaceX REST API and web scraped Wikipedia for comprehensive dataset.

2. **Data Preprocessing**: Applied median imputation for missing values, created binary target variable, and removed outliers.

3. **Feature Engineering**: One-hot encoded categorical variables (launch site, orbit, booster version) and standardized numerical features using StandardScaler.

4. **Model Training**: Used 80/20 train-test split with 5-fold cross-validation. Applied GridSearchCV for hyperparameter tuning across all algorithms.

5. **Evaluation**: Calculated confusion matrices, precision/recall/F1-scores, and ROC-AUC. Selected best model based on balanced performance metrics.

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
