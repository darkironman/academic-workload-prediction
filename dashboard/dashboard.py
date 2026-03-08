# Import required libraries
# streamlit → to build the web dashboard
# mysql.connector → to connect Python with MySQL database
# pandas → to work with data
# pickle → to load saved ML model
# os → to work with files/folders
# matplotlib → for pie chart visualization

import streamlit as st
import mysql.connector
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt


# Set page configuration (title and layout)
st.set_page_config(
    page_title="Academic Workload Monitoring System",
    layout="wide"
)

# Main dashboard title and description
st.title(" Academic Workload Monitoring System")
st.caption("Real-time academic tracking and workload prediction")
st.divider()


# Function to connect to MySQL database
# This will be used whenever we need database access
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="YOUR_PASSWORD", 
        database="academic_workload_db"
    )


# Connect to database
conn = get_connection()

# Fetch all academic activity data (latest week first)
activity_df = pd.read_sql(
    "SELECT * FROM academic_activity ORDER BY week_no DESC",
    conn
)

# Fetch latest model performance details
metadata_df = pd.read_sql(
    "SELECT * FROM model_metadata ORDER BY trained_on DESC LIMIT 1",
    conn
)

# Close connection after fetching data
conn.close()


st.subheader(" Overview")

# Create 4 metric columns
c1, c2, c3, c4 = st.columns(4)

# Display summary statistics
c1.metric("Total Weeks", len(activity_df))
c2.metric("Avg Assignments", round(activity_df["assignments_count"].mean(), 2))
c3.metric("Avg Quizzes", round(activity_df["quizzes_count"].mean(), 2))
c4.metric("Latest Week", activity_df["week_no"].max())

st.divider()

st.subheader(" Academic Analytics")

col1, col2 = st.columns(2)


# Left side: Workload distribution pie chart
with col1:
    st.markdown("### Workload Distribution")

    # Count how many Low / Medium / High
    dist = activity_df["workload_label"].value_counts()

    # Create pie chart
    fig, ax = plt.subplots()
    ax.pie(dist, labels=dist.index, autopct="%1.1f%%")
    ax.set_title("Workload Levels")

    st.pyplot(fig)


# Right side: Workload trend over weeks
with col2:
    st.markdown("### Workload Trend")

    # Sort data by week number
    trend_df = activity_df.sort_values("week_no")

    # Convert workload labels into numbers for plotting
    mapping = {"Low": 1, "Medium": 2, "High": 3}
    trend_df["workload_numeric"] = trend_df["workload_label"].map(mapping)

    # Show line chart
    st.line_chart(
        trend_df.set_index("week_no")["workload_numeric"]
    )
    st.markdown("### Exam Proximity Impact")

exam_impact = activity_df.groupby("exam_proximity")["workload_label"].value_counts().unstack()

exam_impact.index = ["No Exam Nearby", "Exam Nearby"]

fig, ax = plt.subplots(figsize=(4,3))   

exam_impact.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    colormap="viridis"
)

ax.set_xlabel("Exam Status")
ax.set_ylabel("Count")
ax.set_title("Exam vs Workload")

plt.xticks(rotation=0)

st.pyplot(fig, use_container_width=False)   

st.divider()

st.subheader(" Predict Upcoming Workload")

# Load latest trained model from models folder
model_files = [
    os.path.join("models", f)
    for f in os.listdir("models")
    if f.endswith(".pkl")
]

# If no model exists, stop the app
if not model_files:
    st.error("No trained model found.")
    st.stop()

# Get most recent model
latest_model = max(model_files, key=os.path.getmtime)

# Load model using pickle
with open(latest_model, "rb") as f:
    model = pickle.load(f)


# Create input fields for user
col1, col2, col3, col4 = st.columns(4)

with col1:
    assignments = st.number_input("Assignments", 0, 10, 2)

with col2:
    quizzes = st.number_input("Quizzes", 0, 5, 1)

with col3:
    exam = st.selectbox("Exam Nearby?", [0, 1])

with col4:
    prev_workload = st.selectbox("Previous Workload", [1, 2, 3])


# When user clicks Predict button
if st.button("Predict Workload"):

    # Create input dataframe for model
    input_df = pd.DataFrame([{
        "assignments_count": assignments,
        "quizzes_count": quizzes,
        "exam_proximity": exam,
        "previous_workload": prev_workload
    }])

    # Get prediction
    prediction = model.predict(input_df)[0]

    # Get probability confidence
    probabilities = model.predict_proba(input_df)[0]
    confidence = round(max(probabilities) * 100, 2)

    # Show prediction result
    if prediction == "High":
        st.error(f"Predicted Workload: {prediction}")
    elif prediction == "Medium":
        st.warning(f"Predicted Workload: {prediction}")
    else:
        st.success(f"Predicted Workload: {prediction}")

    # Show confidence score
    st.info(f"Model Confidence: {confidence}%")

    # Save prediction into database
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO weekly_predictions (week_no, predicted_workload)
        VALUES (%s, %s)
        """,
        (int(activity_df["week_no"].max()) + 1, str(prediction))
    )

    conn.commit()
    conn.close()

st.divider()

st.subheader(" Model Performance")

if not metadata_df.empty:

    # Display model metrics
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", metadata_df["accuracy"][0])
    m2.metric("Precision", metadata_df["precision_score"][0])
    m3.metric("Recall", metadata_df["recall_score"][0])

    # Calculate F1 score manually
    f1 = round(
        (2 * metadata_df["precision_score"][0] * metadata_df["recall_score"][0]) /
        (metadata_df["precision_score"][0] + metadata_df["recall_score"][0] + 0.0001),
        2
    )

    m4.metric("F1 Score", f1)

    st.caption(f"Last Trained On: {metadata_df['trained_on'][0]}")

else:
    st.info("No model performance data available.")


# Show accuracy trend over time
conn = get_connection()
history_df = pd.read_sql(
    "SELECT trained_on, accuracy FROM model_metadata ORDER BY trained_on ASC",
    conn
)
conn.close()

if not history_df.empty:
    st.markdown("### Accuracy Over Time")
    st.line_chart(history_df.set_index("trained_on")["accuracy"])

st.divider()

st.subheader(" Recent Predictions")

conn = get_connection()
pred_df = pd.read_sql(
    "SELECT * FROM weekly_predictions ORDER BY week_no DESC LIMIT 10",
    conn
)
conn.close()

# Display last 10 predictions
st.dataframe(pred_df, use_container_width=True)


