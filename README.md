# **RASP: Road Accident Severity Prediction**

### **1. Project Selection**

**Problem Statement:**
- Predict the severity of road accidents (Slight, Serious, Fatal) using environmental, human, and vehicle-related factors. (from road, weather, driver, and vehicle features.)

**Why chosen:**

- Road safety is a major global challenge.
- Fatal accidents are rare but devastating. Predicting severity can help with preventive policies.
- Identifying risk factors can help policymakers design better interventions.
- Dataset is rich, real-world, and publicly available.

Type of ML Task:

- Classification (multi-class).

**Who benefits?**

- Transport authorities (NTSA, UK Dept of Transport)
- Emergency responders
- Road safety planners


### **2. Data Collection & Understanding**

Source: UK STATS19 Road Safety Dataset


Tables used:
- Collisions (road & environment factors)
- Casualties (human factors)
- Vehicles (vehicle details)

**Exploration:**

- Data types: mix of categorical (weather, light, road surface) and numerical (speed limit, age).

- Missing values present in some fields.

- Severe class imbalance (Fatal ≪ Slight).

- Outliers: extreme ages, invalid speed values.

**EDA visualizations:**

- Severity distribution.

- Accidents by time of day.

- Correlation between road conditions & severity.

### **3. Data Preprocessing**

Handle missing values (impute with mode/median, “Unknown” for categories).

Encode categorical variables (One-Hot or Label encoding).

Normalize/standardize numerical features if required.

Address imbalanced target (use SMOTE oversampling + class weights).

Split into Train (70%) / Test (30%) sets.

### **4. Modeling**

- Baseline models: Logistic Regression, Decision Tree.

- Advanced models: Random Forest, XGBoost.

- Compare performance across models.

- Use GridSearchCV / RandomizedSearchCV for hyperparameter tuning.

- Select best-performing model (tradeoff between accuracy & interpretability).

### **5. Evaluation**

**Metrics:**
- Accuracy
- Precision, Recall, F1 (macro average for multi-class)
- Confusion matrix

**Model diagnostics:**

- Validation curves (e.g., depth in trees).
- Learning curves (detect underfitting/overfitting).
- Feature importance: Identify top contributors to severity (e.g., weather, light, speed limit).

### **6. Error Analysis**

Confusion matrix to see which severities are most confused.

Fatal vs Serious often misclassified (due to low fatal samples).

Errors higher during rare conditions (e.g., fog).

**Reasons:**

Imbalanced classes (few fatal samples).

Some features are noisy or incomplete.

**Improvements:**

Collect more balanced data.

Engineer features (rush hour, weekend/weekday, rural/urban).

Try ensemble models.

### **7. Model Interpretation**

Tree-based feature importance: e.g., road type, speed limit, light conditions.

SHAP values: show how individual features contribute to predictions.

Plain explanation:

“Accidents on wet roads at night are more likely to be serious/fatal.”

“Higher speed limits correlate with more severe accidents.”

### **8. Deployment**

Build a Streamlit app:

Input accident conditions → Output predicted severity.

Show top contributing features.

Alternatively, deliver a documented Jupyter Notebook.

### **9. Project Report Structure**

- Title & Abstract – Road Accident Severity Prediction using Machine Learning

- Problem Statement – Why this matters (road safety)

- Data Collection & Understanding – STATS19 dataset overview, EDA

- Data Preprocessing – Cleaning, encoding, balancing

- Modeling Approach – Models tried, tuning process

- Results & Evaluation – Metrics, confusion matrix, learning/validation curves

- Error Analysis – Where model struggles and why

- Conclusion & Future Work – Policy insights, improving rare class predictions
