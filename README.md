# **Project: Road Accident Severity Prediction**

Goal: Build a supervised classification model to predict the severity of a road accident (Slight, Serious, or Fatal) using features such as weather, road conditions, time of day, vehicle type, etc.

1. Problem Definition
Business context: Transport authorities want to identify high-risk conditions for fatal accidents to improve safety policies.
ML framing: Multi-class classification (target variable = accident severity).

2. Data Acquisition
Dataset options:
UK Road Safety Data (Data.gov.uk)
 – yearly CSV files with thousands of accidents.
Kenya Open Data
 – check NTSA accident reports (smaller dataset, but local relevance).
Each record typically has:
Categorical features: road surface (dry/wet), weather, light conditions, vehicle type.
Numerical features: number of vehicles involved, number of casualties, age of driver.
Target: Accident severity (slight, serious, fatal).

3. Data Preprocessing
- Handle missing values (e.g., unknown weather → "Unknown").
- Convert categorical variables (e.g., weather = sunny/rain/snow) → One-hot encoding or label encoding.
- Normalize/standardize numerical variables (optional, especially if using distance-based models).
- Handle class imbalance (fatal accidents are much fewer than slight ones) using:
- Oversampling (SMOTE)
- Class weights in models

4. Exploratory Data Analysis (EDA)
- Accident frequency by time of day, day of week, weather, and road conditions.
- Accident severity distribution (imbalanced!).
- Correlation heatmaps for numeric features.
- Feature importance preview (Random Forest/XGBoost).

5. Feature Engineering

6. Model Selection

7. Model Evaluation

8. Hyperparameter Tuning
- GridSearchCV or RandomizedSearchCV 

9. Model Interpretation

10. Deployment

### Deliverables:

Clean dataset.
- EDA visualizations.
- Model training notebook (with multiple models + tuned final model).
- Evaluation report (with confusion matrix, precision/recall/F1).
- Deployment app or dashboard.
