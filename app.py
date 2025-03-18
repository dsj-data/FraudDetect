from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# initialize flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('fraud_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    step = float(request.form['step'])
    amount = float(request.form['amount'])
    oldbalanceOrg = float(request.form['oldbalanceOrg'])
    newbalanceOrig = float(request.form['newbalanceOrig'])
    oldbalanceDest = float(request.form['oldbalanceDest'])
    newbalanceDest = float(request.form['newbalanceDest'])
    transaction_type = request.form['transaction_type']
        
    print(f"Input values: {step}, {amount}, {oldbalanceOrg}, {newbalanceOrig}, {oldbalanceDest}, {newbalanceDest}, {transaction_type}")
     
    # Convert transaction type to one-hot encoding
    transaction_types = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    type_encoding = {t: 1 if t == transaction_type else 0 for t in transaction_types}
    
    print(f"Encoded transaction types: {type_encoding}")

    # Prepare data as DataFrame
    df = pd.DataFrame([{
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        **type_encoding
    }])

    print(f"Prepared DataFrame: {df}")

    # Preprocess the input features (handle scaling, one-hot encoding, etc.)
    df['bal_diff_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['bal_diff_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Predict using the model
    prediction = model.predict(df)[0]
    print(f"Prediction: {prediction}")
    
    # Return result
    prediction_message = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
    return render_template('index.html', prediction_message=prediction_message)




if __name__ == '__main__':
    app.run(debug=True)

