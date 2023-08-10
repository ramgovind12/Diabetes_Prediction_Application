from flask import Flask, render_template, request
import joblib
app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    features = [float(x) for x in request.form.values()]
    
    # Apply the same scaling as during training
    features_scaled = scaler.transform([features])
    
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        result = 'Diabetic'
    else:
        result = 'Non-Diabetic'

    return render_template('index.html', prediction_text=f'The patient is likely {result}.')

if __name__ == '__main__':
    app.run(debug=True)