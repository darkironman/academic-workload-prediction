# dashboard.py
# This file displays live academic workload data,
# allows users to predict upcoming workload,
# and shows model performance using a dashboard.

import streamlit as st
import mysql.connector
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Academic Workload Dashboard", layout="wide")
st.title(" Academic Workload Prediction Dashboard")

# Function to connect to MySQL database
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="yashironman1@",
        database="academic_workload_db"
    )
# Load academic activity and model metadata from database
conn = get_connection()

activity_df = pd.read_sql(
    "SELECT * FROM academic_activity ORDER BY week_no DESC",
    conn
)

metadata_df = pd.read_sql(
    "SELECT * FROM model_metadata ORDER BY trained_on DESC LIMIT 1",
    conn
)

conn.close()

# Display recent academic activity in a scrollable table
st.subheader(" Recent Academic Activity (Live Data)")
st.dataframe(activity_df, use_container_width=True, height=300)

# Show basic statistics about the academic data
st.subheader(" Data Overview")
c1, c2, c3 = st.columns(3)

c1.metric("Total Weeks", len(activity_df))
c2.metric("Avg Assignments", round(activity_df["assignments_count"].mean(), 2))
c3.metric("Avg Quizzes", round(activity_df["quizzes_count"].mean(), 2))

st.bar_chart(activity_df.groupby("workload_label").size())

# Input fields for predicting next week's workload
st.subheader(" Predict Upcoming Week Workload")

# Load the latest trained model from the models folder
model_files = sorted(
    [f for f in os.listdir("models") if f.endswith(".pkl")],
    reverse=True
)

if not model_files:
    st.error("No trained model found.")
    st.stop()

latest_model = os.path.join("models", model_files[0])

with open(latest_model, "rb") as f:
    model = pickle.load(f)

st.caption(f"Using model: `{model_files[0]}`")

c1, c2, c3, c4 = st.columns(4)

with c1:
    assignments = st.number_input("Assignments", 0, 10, 2)

with c2:
    quizzes = st.number_input("Quizzes", 0, 5, 1)

with c3:
    exam = st.selectbox("Exam Nearby?", [0, 1])

with c4:
    prev_workload = st.selectbox("Previous Workload", [1, 2, 3])

if st.button("Predict Workload", key="predict_btn"):
# Convert user inputs into DataFrame for model prediction
    input_df = pd.DataFrame([{
        "assignments_count": assignments,
        "quizzes_count": quizzes,
        "exam_proximity": exam,
        "previous_workload": prev_workload
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Workload Level: **{prediction}**")
# Store prediction in database for future reference
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO weekly_predictions (week_no, predicted_workload) VALUES (%s, %s)",
        (int(activity_df["week_no"].max()) + 1, str(prediction))
    )

    conn.commit()
    conn.close()
# Display latest model performance metrics
st.subheader(" Model Performance")

if not metadata_df.empty:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", metadata_df["accuracy"][0])
    c2.metric("Precision", metadata_df["precision_score"][0])
    c3.metric("Recall", metadata_df["recall_score"][0])

    f1 = round(
        (2 * metadata_df["precision_score"][0] * metadata_df["recall_score"][0]) /
        (metadata_df["precision_score"][0] + metadata_df["recall_score"][0] + 0.0001),
        2
    )

    c4.metric("F1 Score", f1)

    st.caption(f"Model trained on: {metadata_df['trained_on'][0]}")
else:
    st.info("No model metrics available.")


