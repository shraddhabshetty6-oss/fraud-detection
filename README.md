
# 🔍 Fraud Detection Using Machine Learning

## 📌 Project Overview

This project focuses on detecting potentially fraudulent transactions using Python and Machine Learning.

The project analyzes transaction data from an Excel dataset and identifies suspicious transactions using predefined rules based on transaction price, quantity, and order status.

Two machine learning algorithms are implemented and compared:

- Logistic Regression
- Random Forest Classifier

The project also uses data preprocessing techniques and SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced data.

## 🎯 Objectives

- Identify potentially suspicious transactions.
- Perform data preprocessing and cleaning.
- Create fraud labels using the Interquartile Range (IQR) method.
- Handle categorical and numerical features.
- Apply SMOTE to address class imbalance.
- Train and compare two machine learning models.
- Evaluate model performance using Precision, Recall, and ROC-AUC.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Imbalanced-learn | SMOTE and imbalanced learning pipeline |
| Excel | Dataset format |

## 📂 Project Structure

```text
fraud-detection/
│
├── Dataset for Data Analytics (3) (1).xlsx
├── fraud_detection.py
└── README.md
```

> Replace `fraud_detection.py` with your actual Python filename if it is different.

## 📊 Dataset

The project uses an Excel dataset containing transaction-related information.

Important columns used in the project include:

- `TotalPrice` – Total transaction price.
- `Quantity` – Quantity of items in the transaction.
- `OrderStatus` – Status of the order.
- `OrderID` – Order identifier.
- `CustomerID` – Customer identifier.
- `TrackingNumber` – Shipment tracking information.
- `Date` – Transaction date.

The dataset is loaded using Pandas:

```python
data = pd.read_excel(dataset_path)
```

## 🔎 Fraud Label Creation

The project creates a `Fraud` column using rule-based detection.

A transaction is labeled as potentially fraudulent (`1`) when at least one of the following conditions is met:

1. `TotalPrice` is greater than the upper IQR threshold.
2. `Quantity` is greater than the upper IQR threshold.
3. `OrderStatus` is equal to `cancelled`.

Otherwise, the transaction is labeled as `0`.

### IQR Method

The Interquartile Range (IQR) is calculated as:

```text
IQR = Q3 - Q1
```

The upper threshold is:

```text
Upper Bound = Q3 + 1.5 × IQR
```

This rule is used to identify unusually high transaction prices and quantities.

> Note: These are rule-based suspicious transaction labels, not independently verified fraud cases.

## ⚙️ Data Preprocessing

The following preprocessing techniques are used:

### Numerical Features

- Missing values are filled using the median.
- Features are scaled using `StandardScaler`.

### Categorical Features

- Missing values are filled using the most frequent value.
- Categorical values are converted into numerical form using `OneHotEncoder`.

### Removed Columns

The following columns are excluded from model training:

- Fraud
- OrderID
- CustomerID
- TrackingNumber
- Date

## 🧠 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as a classification model to predict whether a transaction belongs to the suspicious or non-suspicious class.

The model is trained with:

- Data preprocessing pipeline.
- SMOTE oversampling.
- Logistic Regression classifier.

### 2. Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees for classification.

In this project, the model is configured with:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

The Random Forest model is trained using the same preprocessing and SMOTE pipeline.

## ⚖️ Handling Class Imbalance

The project uses **SMOTE (Synthetic Minority Over-sampling Technique)**.

SMOTE is applied after preprocessing and before model training to address class imbalance in the training data.

```python
("smote", SMOTE(random_state=42))
```

The SMOTE step is included in an imbalanced-learn pipeline.

## 📈 Model Evaluation

The following evaluation metrics are used:

### Precision

Measures the proportion of transactions predicted as fraudulent that are actually in the fraud-labeled class.

### Recall

Measures the proportion of fraud-labeled transactions that the model correctly identifies.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Model Comparison

The project compares:

| Model | Precision | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | Calculated during execution | Calculated during execution | Calculated during execution |
| Random Forest | Calculated during execution | Calculated during execution | Calculated during execution |

The actual values are displayed when the Python program runs.

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shraddhabshetty6-oss/fraud-detection.git
```

### 2. Navigate to the Project Directory

```bash
cd fraud-detection
```

### 3. Install Required Libraries

```bash
pip install pandas numpy scikit-learn imbalanced-learn openpyxl
```

### 4. Run the Project

```bash
python fraud_detection.py
```

Make sure the Excel dataset is located in the same directory as the Python script.

## 💻 Project Workflow

```text
Load Excel Dataset
        ↓
Create Fraud Labels
        ↓
Remove Unnecessary Columns
        ↓
Identify Numerical and Categorical Features
        ↓
Preprocess Data
        ↓
Split Data into Training and Testing Sets
        ↓
Apply SMOTE
        ↓
Train Logistic Regression
        ↓
Train Random Forest
        ↓
Evaluate Models
        ↓
Compare Results
```

## 🎓 Learning Outcomes

Through this project, I learned:

- Python-based data analysis.
- Working with Excel datasets.
- Data preprocessing techniques.
- Rule-based fraud labeling.
- Handling imbalanced datasets using SMOTE.
- Implementing Logistic Regression and Random Forest.
- Evaluating classification models.

## 🔮 Future Enhancements

- Use verified fraud labels when available.
- Add more transaction-related features.
- Explore additional machine learning algorithms.
- Create data visualizations.
- Develop an interactive fraud detection dashboard.
- Improve fraud detection through better feature engineering.
- Evaluate models using additional classification metrics.

## 👩‍💻 Author

**Shraddha B Shetty**

GitHub: https://github.com/shraddhabshetty6-oss

## 📄 License

This project is created for educational and learning purposes.
