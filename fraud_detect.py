## Install/import required libraries------------------------ 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
#------------------------------------------------------


#  explore the data
df = pd.read_csv("C:/Users/Winni/.kaggle/Fraud.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().round(3))

# check for missing vals
df.isna().sum()

# check the distribution of fraud cases
# see how imbalanced the dataset is
df['isFraud'].value_counts(normalize=True) *100
# shows that only 13% is fraud...imbalanced
# may need oversample fraud
# or undersample non-fraud

df['isFlaggedFraud'].value_counts(normalize=True) * 100
# less than 0.0002% is flagged fraud...not useful


#---------------------------------------------------
# drop unnecessary columns
unneeded = ['nameOrig','nameDest','isFlaggedFraud']
df.drop(columns= unneeded, inplace=True)

#separate features and target
X = df.drop(columns=['isFraud'])
y = df['isFraud']


#---------------------------------------------------
# handle class imbalance using SMOTE
# synethetic minority oversampling tech

#sep num and cate features
X_num = X.select_dtypes(exclude=['object'])
X_cat = X.select_dtypes(include=['object'])

#apply smote to num
smote = SMOTE(random_state = 42)
X_num_res, y_res = smote.fit_resample(X_num,y)

#recombine num and cat
X_res_comb = pd.concat([pd.DataFrame(X_num_res,columns=X_num.columns),
                        X_cat.reset_index(drop=True)], axis = 1)


#---------------------------------------------------
# type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
# one-hot encoding for type
X_res_fin = pd.get_dummies(X_res_comb, columns=['type'], drop_first = True)



#---------------------------------------------------
# scale
scaler = StandardScaler()
# Fit and transform the features
scale_col = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X_res_fin[scale_col] = scaler.fit_transform(X_res_fin[scale_col])



#---------------------------------------------------
#check outliers for balances
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Detect outliers in fraudulent transactions
transaction_types = [col for col in X_res_fin.columns if 'type_' in col]
print(transaction_types)

for types in transaction_types:
    # Check outliers for sender and recipient balances
    for balance_column in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        outliers = detect_outliers_iqr(X_res_fin[X_res_fin[types] == 1], balance_column)
        print(f"Outliers for {balance_column} - {types}: {len(outliers)}")
    print('-'*20)

# feature engineering for balances
# balance difference
X_res_fin['bal_diff_orig'] = X_res_fin['newbalanceOrig'] - X_res_fin['oldbalanceOrg']
X_res_fin['bal_diff_dest'] = X_res_fin['newbalanceDest'] - X_res_fin['oldbalanceDest']



#---------------------------------------------------
#model & eval
# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_res_fin, y_res, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# predict 
y_pred = model.predict(X_test)

#eval model
# Accuracy, Precision, Recall, F1-Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ROC AUC score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Print all evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print('ROC AUC:', roc_auc)


#--------------------------------------------
#for flask app
# Save the model to a file
joblib.dump(model, 'fraud_model.pkl')


