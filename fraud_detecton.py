import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ==========================================
# 1. LOAD DATASET
# ==========================================

dataset_path = Path(__file__).resolve().parent / "Dataset for Data Analytics (3) (1).xlsx"
data = pd.read_excel(dataset_path)

print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# ==========================================
# 2. CREATE FRAUD LABEL
# ==========================================

# Calculate unusual values using IQR

Q1 = data["TotalPrice"].quantile(0.25)
Q3 = data["TotalPrice"].quantile(0.75)
IQR = Q3 - Q1

upper_price = Q3 + 1.5 * IQR

quantity_Q1 = data["Quantity"].quantile(0.25)
quantity_Q3 = data["Quantity"].quantile(0.75)
quantity_IQR = quantity_Q3 - quantity_Q1

upper_quantity = quantity_Q3 + 1.5 * quantity_IQR


# Mark suspicious transactions

data["Fraud"] = (
    (data["TotalPrice"] > upper_price) |
    (data["Quantity"] > upper_quantity) |
    (data["OrderStatus"].astype(str).str.lower() == "cancelled")
).astype(int)


print("\nFraud distribution:")
print(data["Fraud"].value_counts())


# ==========================================
# 3. REMOVE UNNECESSARY COLUMNS
# ==========================================

X = data.drop(
    ["Fraud", "OrderID", "CustomerID", "TrackingNumber", "Date"],
    axis=1
)

y = data["Fraud"]


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==========================================
# 7. LOGISTIC REGRESSION + SMOTE
# ==========================================

logistic_model = ImbPipeline([
    ("preprocessing", preprocessor),

    ("smote", SMOTE(random_state=42)),

    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

logistic_model.fit(X_train, y_train)

logistic_prediction = logistic_model.predict(X_test)

logistic_probability = (
    logistic_model.predict_proba(X_test)[:, 1]
)


# ==========================================
# 8. LOGISTIC REGRESSION RESULTS
# ==========================================

logistic_precision = precision_score(
    y_test,
    logistic_prediction,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_prediction,
    zero_division=0
)

logistic_roc_auc = roc_auc_score(
    y_test,
    logistic_probability
)

print("\n================================")
print("LOGISTIC REGRESSION")
print("================================")

print("Precision:", logistic_precision)
print("Recall:", logistic_recall)
print("ROC-AUC:", logistic_roc_auc)


# ==========================================
# 9. RANDOM FOREST + SMOTE
# ==========================================

random_forest_model = ImbPipeline([
    ("preprocessing", preprocessor),

    ("smote", SMOTE(random_state=42)),

    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

random_forest_model.fit(X_train, y_train)

random_forest_prediction = (
    random_forest_model.predict(X_test)
)

random_forest_probability = (
    random_forest_model.predict_proba(X_test)[:, 1]
)


# ==========================================
# 10. RANDOM FOREST RESULTS
# ==========================================

random_forest_precision = precision_score(
    y_test,
    random_forest_prediction,
    zero_division=0
)

random_forest_recall = recall_score(
    y_test,
    random_forest_prediction,
    zero_division=0
)

random_forest_roc_auc = roc_auc_score(
    y_test,
    random_forest_probability
)

print("\n================================")
print("RANDOM FOREST")
print("================================")

print("Precision:", random_forest_precision)
print("Recall:", random_forest_recall)
print("ROC-AUC:", random_forest_roc_auc)


# ==========================================
# 11. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],

    "Precision": [
        logistic_precision,
        random_forest_precision
    ],

    "Recall": [
        logistic_recall,
        random_forest_recall
    ],

    "ROC-AUC": [
        logistic_roc_auc,
        random_forest_roc_auc
    ]
})


print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(results.to_string(index=False))