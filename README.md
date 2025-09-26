# Fraud Detection Model (PaySim Dataset)

## Overview
Built a fraud detection model using the **PaySim synthetic dataset** (600K+ transaction records).  
Goal: Detect fraudulent transactions with high recall while minimizing false positives. Then provide a web-based prediction system.

--------------------------
## Approach
1. Data preprocessing (handling imbalance, feature engineering).  
2. Model comparison: Logistic Regression, Random Forest, XGBoost.  
3. Applied **SMOTE** + hyperparameter tuning for XGBoost.
4. Deployed using Flask (for real-time prediction)
   
--------------------------
## Data Processing Workflow

**1) Download & Load Data**
- Data is fetched from Kaggle using load.py.
- fraud_detect.py reads the dataset (Fraud.csv).

**2) Exploratory Data Analysis (EDA)**
- Basic statistics & missing value check
- Distribution of fraud vs. legitimate transactions

**3) Feature Engineering**
- Unnecessary columns dropped: nameOrig, nameDest, isFlaggedFraud
- Transaction type is one-hot encoded
- New features created: bal_diff_orig, bal_diff_dest
- StandardScaler applied to numerical features

**4) Handling Class Imbalance**
- SMOTE (Synthetic Minority Over-sampling Technique) used to balance fraud cases

**5) Model Training & Evaluation**
- RandomForestClassifier trained on resampled data
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

**6) Deploy Model with Flask**
- app.py loads the trained model (fraud_model.pkl)
- Receives transaction details via web form (index.html)
- Displays fraud prediction results

--------------------------
## Results
- XGBoost: **92% accuracy, 87% recall, AUC = 0.95**  
- Outperformed baseline Logistic Regression (recall = 63%).  

--------------------------
## Business Impact
- Improved fraud detection performance → potential to save millions in fraudulent losses.

--------------------------
## Next Steps
- Use Deep Learning models (e.g., LSTM, CNN) for better fraud detection
- Implement real-time streaming detection using Kafka
- Deploy to AWS/GCP for cloud-based fraud detection





--------------------------
**Original Data Dictionary:**

step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).

type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.

amount - amount of the transaction in local currency.

nameOrig - customer who started the transaction

oldbalanceOrg - initial balance before the transaction

newbalanceOrig - new balance after the transaction

nameDest - customer who is the recipient of the transaction

oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).

newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).

isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.

isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

