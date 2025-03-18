# Fraud Detection System
This is a machine learning-based fraud detection system that identifies fraudulent transactions using a RandomForest classifier.
The model is trained on Kaggle's Fraudulent Transactions Dataset and deployed using Flask.

--------------------------
**Project Overview**
Fraudulent financial transactions can cause significant losses to businesses and individuals.
This project aims to detect fraudulent transactions using machine learning and provide a web-based prediction system.

--------------------------
**Key Features**
- Kaggle dataset integration
- Data preprocessing & feature engineering
- Handling imbalanced data with SMOTE
- Fraud detection using RandomForestClassifier
- Flask web app for real-time prediction

--------------------------
**Data Processing Workflow**
1) Download & Load Data
Data is fetched from Kaggle using load.py.
fraud_detect.py reads the dataset (Fraud.csv).

2) Exploratory Data Analysis (EDA)
Basic statistics & missing value check
Distribution of fraud vs. legitimate transactions

3) Feature Engineering
Unnecessary columns dropped: nameOrig, nameDest, isFlaggedFraud
Transaction type is one-hot encoded
New features created: bal_diff_orig, bal_diff_dest
StandardScaler applied to numerical features

4) Handling Class Imbalance
SMOTE (Synthetic Minority Over-sampling Technique) used to balance fraud cases

5) Model Training & Evaluation
RandomForestClassifier trained on resampled data
Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

6) Deploy Model with Flask
app.py loads the trained model (fraud_model.pkl)
Receives transaction details via web form (index.html)
Displays fraud prediction results

--------------------------
**Improvements & Next Steps**
Use Deep Learning models (e.g., LSTM, CNN) for better fraud detection
Implement real-time streaming detection using Kafka
Deploy to AWS/GCP for cloud-based fraud detection

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
