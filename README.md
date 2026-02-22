Academic Workload Prediction System
Project Overview

This project predicts the academic workload level (Low / Medium / High) for upcoming weeks using past academic activity data.
The system helps students and faculty understand workload trends in advance so they can plan assignments, quizzes, and exam preparation better.

The project uses:
Live data stored in MySQL
A simple machine learning classification model
An interactive Streamlit dashboard

Problem Statement
Students often face sudden increases in academic workload due to multiple assignments, quizzes, and exams happening close together.
There is no clear way to estimate how heavy the upcoming academic week will be.
This project solves that problem by:
Collecting weekly academic activity data

Learning patterns from past data
Predicting the workload level for the next week
Showing results and trends on a dashboard

Technologies Used

Python
MySQL
Pandas
Scikit-learn
Streamlit
Logistic Regression 

Database Structure
academic_activity
Stores weekly academic activity data.

Column Name	Description
record_id	Primary key
week_no	Academic week number
assignments_count	Number of assignments
quizzes_count	Number of quizzes
exam_proximity	1 = exam nearby, 0 = no exam
previous_workload	Previous week workload (1-Low, 2-Medium, 3-High)
workload_label	Actual workload label
created_at	Timestamp
model_metadata

Stores model performance and version details.

Column Name	Description
model_id	Primary key
model_version	Model file version
accuracy	Accuracy score
precision_score	Precision score
recall_score	Recall score
trained_on	Training timestamp
weekly_predictions

Stores predicted workload history.

Column Name	Description
prediction_id	Primary key
week_no	Predicted week number
predicted_workload	Predicted workload level
prediction_date	Timestamp
Machine Learning Model
Model Used

Logistic Regression

This model is used as a classification model because:
It is simple and easy to understand
It works well for multi-class prediction
It is suitable for small academic datasets

Input Features
Assignments count
Quizzes coun
Exam proximity
Previous workload

Output Labels

1 → Low
2 → Medium
3 → High

Model Training Process

Data is fetched directly from MySQL
Rows with missing workload labels are removed
Data is split into training and testing sets
Model is trained using Logistic Regression
Performance metrics calculated:

Accuracy

Precision
Recall
F1 Score

Trained models are saved with timestamps
Model performance is stored in the database

Live Data Simulation

A background script automatically inserts new weekly academic data every minute to simulate live data.
This demonstrates:
Continuous data collection

Real-time updates
Model retraining capability
Dashboard refresh with new data

Dashboard Features
Scrollable table showing recent academic activity
Basic data statistics (averages and counts)
Workload distribution visualization

Upcoming week workload prediction
Model performance metrics (Accuracy, Precision, Recall, F1)
Prediction history stored in the database
Automatically uses the latest trained model

How to Run the Project
1. Start MySQL Server

Make sure MySQL is running and the database is created.

2. Run Live Data Generator
python scripts/add_weekly_data.py

3. Train the Model
python scripts/train_model.py

4. Run the Dashboard
streamlit run dashboard/dashboard.py
