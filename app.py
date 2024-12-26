from flask import Flask, request, render_template
import pickle
import numpy as np

# Flask application setup
app = Flask(__name__)

# Load the model and encoders
model = pickle.load(open('placement_model.pkl', 'rb'))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form

        # Prepare features for prediction
        features = [
            float(input_data['CPI']),
            float(input_data['English_Speaking_Skills']),
            float(input_data['Grammar_Skills']),
            float(input_data['Coding_Skills']),
            int(input_data['Total_No_of_Projects']),
            encoders['categorical']['Project_Level'].transform([input_data['Project_Level']])[0],
            encoders['categorical']['Internship_Level'].transform([input_data['Internship_Level']])[0],
            encoders['categorical']['Certification_Level'].transform([input_data['Certification_Level']])[0],
            int(input_data['Total_No_of_Internships']),
            int(input_data['Total_No_of_Certifications']),
            float(input_data['Programming_Skills']),
            encoders['categorical']['Field_of_Study'].transform([input_data['Field_of_Study']])[0],
            int(input_data['Total_No_of_LeetCode_Questions_Solved']),
            int(input_data['Total_No_of_Questions_Other_Platform'])
        ]

        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'
        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
