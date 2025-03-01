import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('student_prediction.csv')

# Function to predict the SUMM value based on input data
def predict_summ(data):
    try:
        # Extract input values from the form data
        input_values = [
            int(data['age']),
            int(data['gender']),
            int(data['study_hrs']),
            int(data['read_freq']),
            int(data['impact']),
            int(data['attend']),
            int(data['prep_study']),
            int(data['prep_exam']),
            int(data['notes']),
            int(data['listens']),
            int(data['cuml_gpa']),
            int(data['exp_gpa'])
        ]
        
        # Select relevant columns from the dataset
        relevant_columns = [
            'AGE', 'GENDER', 'STUDY_HRS', 'READ_FREQ', 'IMPACT', 'ATTEND',
            'PREP_STUDY', 'PREP_EXAM', 'NOTES', 'LISTENS', 'CUML_GPA', 'EXP_GPA'
        ]
        
        # Subset the dataset with relevant columns
        relevant_data = df[relevant_columns]
        
        # Calculate distances between input values and dataset records
        distances = euclidean_distances(relevant_data, [input_values])
        
        # Find the index of the record with the smallest distance
        closest_index = distances.argmin()
        
        # Get the SUMM value from the closest matching record
        summ = df.iloc[closest_index]['SUMM']
        
        return summ
    except KeyError as e:
        return f"Error: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/student_prediction', methods=['GET', 'POST'])
def student_prediction():
    if request.method == 'POST':
        # Get form data
        form_data = request.form
        # Predict SUMM
        summ = predict_summ(form_data)
        # Render template with the SUMM result
        return render_template('result.html', summ=summ)
    else:
        # Render the form template for GET requests
        return render_template('student_prediction_form.html')

@app.route('/teaching_evaluation')
def teaching_evaluation():
    return render_template('teaching_evaluation.html')

@app.route('/text_analysis')
def text_analysis():
    return render_template('text_analysis.html')

if __name__ == '__main__':
    app.run(debug=True)
