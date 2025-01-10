from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Read the dataset and preprocess
df = pd.read_csv("C:\\Users\\vipul\\Downloads\\collegePlace.csv")
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
Stream_mapping = {
    'Electronics And Communication': 0,
    'Computer Science': 1,
    'Mechanical': 2,
    'Civil': 3,
    'Information Technology': 4,
    'Electrical': 5
}
df['Stream'] = df['Stream'].map(Stream_mapping)

# Model Training
def train_model(df):
    # Split the data into features (X) and target variable (y)
    X = df[['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'HistoryOfBacklogs']]
    y = df['PlacedOrNot']
    
    # Model Selection: Logistic Regression
    model = LogisticRegression()
    model.fit(X, y)
    
    return model

# Function to predict placement based on user input
def predict_placement(model, user_input):
    # Make prediction
    prediction = model.predict(user_input)
    
    return prediction[0]

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting placement
@app.route('/predict_placement', methods=['POST'])
def predict():
    # Get user input
    user_input = [int(request.form['age']), int(request.form['gender']), int(request.form['stream']), 
                  int(request.form['internships']), float(request.form['cgpa']), int(request.form['hostel']), 
                  int(request.form['backlogs'])]
    
    # Predict placement
    placement_prediction = predict_placement(model, [user_input])
    return render_template('result.html', prediction="Yes" if placement_prediction == 1 else "No")

if __name__ == '__main__':
    # Train the model
    model = train_model(df)
    app.run(debug=True)
