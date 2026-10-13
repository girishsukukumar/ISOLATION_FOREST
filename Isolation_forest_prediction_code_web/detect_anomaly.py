from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import json

app = Flask(__name__)

# CONFIGURATION: Define the path to your saved model file

MODEL_FILE = "stream1_day_visit_count.pkl"

print("⚙️ Initializing Central ML Processing Engine...")
print("⚙️ Loading ...",MODEL_FILE)

# Load the pre-trained model from disk
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model_day_visit_count = pickle.load(f)
    print(f"✅ Pre-trained 1D Isolation Forest loaded successfully from '{MODEL_FILE}'!")
else:
    print(f"❌ Error: Model file '{MODEL_FILE}' not found!")
    print("Please make sure you run your training script first to generate this file.")
    exit(1)
    
MODEL_FILE = "stream2_night_visit_count.pkl"

print("⚙️ Loading ...",MODEL_FILE)

# Load the pre-trained model from disk
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model_night_visit_count = pickle.load(f)
    print(f"✅ Pre-trained 1D Isolation Forest loaded successfully from '{MODEL_FILE}'!")
else:
    print(f"❌ Error: Model file '{MODEL_FILE}' not found!")
    print("Please make sure you run your training script first to generate this file.")
    exit(1)


MODEL_FILE = "stream3_day_visit_duration.pkl"

print("⚙️ Loading ...",MODEL_FILE)

# Load the pre-trained model from disk
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model_day_visit_duration = pickle.load(f)
    print(f"✅ Pre-trained 1D Isolation Forest loaded successfully from '{MODEL_FILE}'!")
else:
    print(f"❌ Error: Model file '{MODEL_FILE}' not found!")
    print("Please make sure you run your training script first to generate this file.")
    exit(1)


MODEL_FILE = "stream4_night_visit_duration.pkl"

print("⚙️ Loading ...",MODEL_FILE)

# Load the pre-trained model from disk
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model_night_visit_duration = pickle.load(f)
    print(f"✅ Pre-trained 1D Isolation Forest loaded successfully from '{MODEL_FILE}'!")
else:
    print(f"❌ Error: Model file '{MODEL_FILE}' not found!")
    print("Please make sure you run your training script first to generate this file.")
    exit(1)


@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly():
    try:
        parsed_data = request.get_json(silent=True)
        if not parsed_data:
            return jsonify({"error": "Missing or invalid JSON payload"}), 400
        print(parsed_data)
        stream_name = parsed_data.get("stream")
        data_value = parsed_data.get("data")

        if (stream_name == "day_visit_count"):
            model = model_day_visit_count
            
        if (stream_name == "night_visit_count"):
            model = model_night_visit_count
            
        if (stream_name == "day_visit_duration"):
            model = model_day_visit_duration
            
        if (stream_name == "night_visit_duration"):
            model = model_night_visit_duration   

        if (stream_name == "BedRoomMovements"):
            model = model_day_visit_count  # for the time being this model used           
            
        if model is None:
            return jsonify({"error": f"Unknown stream '{stream_name}'"}), 400

        X = np.array([[float(data_value)]])
        prediction = model.predict(X)[0]
        raw_score = model.score_samples(X)[0]

        return jsonify({
            "anomaly_detected": bool(prediction == -1),
            "raw_anomaly_score": float(raw_score)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
        
        
if __name__ == '__main__':
    # Listen on all local IP endpoints on port 5000
    app.run(host='0.0.0.0', port=6000, debug=False)


# command line Test code
        
# PS C:\Users\girish> curl.exe -X POST http://localhost:6000/predict_anomaly -H "Content-Type: application/json" -d '{\"stream\": \"day_visit_count\", \"data\": 10.5}'

# $body = @{ stream = "day_visit_count"; data = 10.5 } | ConvertTo-Json
# Invoke-RestMethod -Method Post -Uri "http://localhost:6000/predict_anomaly" -ContentType "application/json" -Body $body

'''
PS C:\Users\girish> curl.exe -X POST http://localhost:6000/predict_anomaly -H "Content-Type: application/json" -d '{\"stream\": \"day_visit_count\", \"data\": 10.5}'
{"anomaly_detected":true,"raw_anomaly_score":-0.7972759015041565}
PS C:\Users\girish> curl.exe -X POST http://localhost:6000/predict_anomaly -H "Content-Type: application/json" -d '{\"stream\": \"night_visit_count\", \"data\": 10.5}'
{"anomaly_detected":true,"raw_anomaly_score":-0.7693729424121065}
PS C:\Users\girish> curl.exe -X POST http://localhost:6000/predict_anomaly -H "Content-Type: application/json" -d '{\"stream\": \"night_visit_duration\", \"data\": 10.5}'
{"anomaly_detected":true,"raw_anomaly_score":-0.7249699760811933}
PS C:\Users\girish> curl.exe -X POST http://localhost:6000/predict_anomaly -H "Content-Type: application/json" -d '{\"stream\": \"day_visit_duration\", \"data\": 10.5}'
{"anomaly_detected":true,"raw_anomaly_score":-0.7618636329766193}
PS C:\Users\girish>
'''
scp -i "C:\Users\girish\Documents\EFY\LightSail\LightsailDefaultKey-ap-south-1.pem" .\detect_anomaly.py ubuntu@3.7.179.252:/home/ubuntu/