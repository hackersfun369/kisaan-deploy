from flask import Flask, request, render_template
import numpy as np
import pickle

# Opening the model
model = pickle.load(open('model.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Retrieve form data
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])  # Correct spelling
    K = int(request.form['Potassium'])     # Correct spelling
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Prepare feature list for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Make prediction
    prediction = model.predict(single_pred)

    # Crop dictionary
    crops_dict = {
        1: "rice",
        2: "maize",
        3: "jute",
        4: "cotton",
        5: "coconut",
        6: "papaya",
        7: "orange",
        8: "apple",
        9: "muskmelon",
        10: "watermelon",
        11: "grapes",
        12: "mango",
        13: "banana",
        14: "pomegranate",
        15: "lentil",
        16: "blackgram",
        17: "mungbean",
        18: "mothbeans",
        19: "pigeonpeas",
        20: "kidneybeans",
        21: "chickpea",
        22: "coffee"
    }

    # Get the predicted crop
    crop_index = prediction[0]  # Corrected variable name from 'predict' to 'prediction'
    if crop_index in crops_dict:
        crop = crops_dict[crop_index]
        result = "{} is the best crop to be cultivated.".format(crop)
    else:
        result = "Sorry, we could not predict the crop to cultivate."

    return render_template('index.html', result=result)

# Python main
if __name__ == "__main__":
    app.run(debug=True)