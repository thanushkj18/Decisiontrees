from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("decision_tree_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Get form values
            gender = int(request.form["gender"])
            part_time_job = int(request.form["part_time_job"])
            absence_days = int(request.form["absence_days"])
            extracurricular = int(request.form["extracurricular_activities"])
            weekly_study = int(request.form["weekly_self_study_hours"])
            career_aspiration = int(request.form["career_aspiration"])

            # Arrange into numpy array (same order as training features)
            features = np.array([[gender, part_time_job, absence_days,
                                  extracurricular, weekly_study, career_aspiration]])

            # Predict result
            prediction = model.predict(features)[0]

            return render_template("index.html", result_text=f"Predicted Result: {prediction}")

        except Exception as e:
            return render_template("index.html", result_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
