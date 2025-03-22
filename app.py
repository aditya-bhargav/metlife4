from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask(__name__)

# Load trained model and encoders
knn = joblib.load("model/knn_classifier.pkl")
encoders = joblib.load("model/encoders.pkl")

@app.route('/')
def home():
    return "Supervised KNN Classifier API 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        df['full_name']=df['first_name']+df['last_name']
        df.drop(columns=['first_name','last_name','ssn','policy_number','policy_start_date','policy_expiry_date','claim_date','hospital_name',	'hospital_location'	,'user_address'],inplace=True)
        if 'Unamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'],inplace=True)
        df.drop('dob', axis=1, inplace=True)



        # Apply encoders to categorical columns
        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[[col]])

        # Predict
        result = knn.predict(df)[0]  # 0: normal, 1: anomaly
        return jsonify({"anomaly": int(result)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
