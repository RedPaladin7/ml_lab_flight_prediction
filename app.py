from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

categories = {
    "airline": ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo',
       'Air_India'],
    "source_city": ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'],
    "departure_time": ['Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night',
       'Late_Night'],
    "stops": ['zero', 'one', 'two_or_more'],
    "arrival_time": ['Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening',
       'Late_Night'],
    "destination_city": ['Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'],
    "class": ['Economy', 'Business'],
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Extract categorical and numerical values
        input_data = {feature: request.form[feature] for feature in categories.keys()}
        input_data["duration"] = float(request.form["duration"])
        input_data["days_left"] = float(request.form["days_left"])

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Preprocess data
        df_processed = preprocessor.transform(df)

        # Predict
        prediction = model.predict(df_processed)[0]
        prediction = round(prediction, 2)

    return render_template("home.html", categories=categories, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
