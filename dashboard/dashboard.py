import streamlit as st
import mysql.connector
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Workload System", layout="wide")

# Theme colors
bg = "#0E1117" 
card = "#161B22" 
text =  "#000000"
border = "#2A2F36" 

# Custom styling
st.markdown(f"""
<style>
body {{ background-color:{bg}; color:{text}; }}
.card {{
    background:{card};
    padding:20px;
    border-radius:12px;
    border:1px solid {border};
    margin-bottom:15px;
}}
</style>
""", unsafe_allow_html=True)

# DB connection function
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="yashironman1@",
        database="academic_workload_db"
    )
# DB connection function
conn = get_connection()
activity_df = pd.read_sql("SELECT * FROM academic_activity ORDER BY week_no DESC", conn)
metadata_df = pd.read_sql("SELECT * FROM model_metadata ORDER BY trained_on DESC", conn)
conn.close()

# Title
st.title(" Academic Workload Monitoring System")
st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs([" System Overview", " Analysis", " Prediction"])

# SYSTEM OVERVIEW
with tab1:

    # INTRO 
    st.markdown("""
    <div class="card">
    <b> Academic Workload Monitoring System</b><br><br>
    This project is designed to track and predict student academic workload 
    using machine learning. It helps students understand how heavy their 
    upcoming week will be based on assignments, quizzes, and exams.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b> How the System Works</b><br><br>
    • Weekly academic data is collected from students<br>
    • Features include assignments, quizzes, exam proximity, and past workload<br>
    • A trained ML model analyzes patterns in this data<br>
    • The system predicts workload as Low, Medium, or High<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <b> Why This Project Matters</b><br><br>
    Many students face stress due to poor workload planning.  
    This system gives early warning so students can manage time better, 
    reduce stress, and improve performance.
    </div>
    """, unsafe_allow_html=True)

    # STATS
    st.subheader(" Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Weeks", len(activity_df))
    c2.metric("Avg Assignments", round(activity_df["assignments_count"].mean(), 2))
    c3.metric("Avg Quizzes", round(activity_df["quizzes_count"].mean(), 2))
    c4.metric("Latest Week", activity_df["week_no"].max())

    # DATA PREVIEW
    #  
    st.subheader(" Dataset Sample")
    st.dataframe(activity_df.head(10), use_container_width=True)

    # MODEL INFO 
    st.subheader(" Model Used")

    st.markdown("""
    <div class="card">
    The system uses two machine learning models:<br><br>
    • Logistic Regression → for stable predictions<br>
    • Decision Tree → for rule-based understanding<br><br>
    The latest trained model is automatically used during prediction.
    </div>
    """, unsafe_allow_html=True)


#  ANALYSIS
with tab2:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        trend_df = activity_df.sort_values("week_no").tail(100)
        mapping = {"Low":1,"Medium":2,"High":3}
        trend_df["num"] = trend_df["workload_label"].map(mapping)

        st.subheader("Workload Trend (Recent)")
        st.line_chart(trend_df.set_index("week_no")["num"])

        st.markdown('</div>', unsafe_allow_html=True)

    # EXAM IMPACT
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        exam = activity_df.groupby("exam_proximity")["workload_label"].value_counts().unstack()
        exam.index = ["No Exam", "Exam Nearby"]

        st.subheader("Exam Impact")
        st.bar_chart(exam)

        st.markdown('</div>', unsafe_allow_html=True)

    #  MODEL GROWTH
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Growth")

    if not metadata_df.empty:
        st.line_chart(metadata_df.set_index("trained_on")["accuracy"])
    else:
        st.info("No training history")

    st.markdown('</div>', unsafe_allow_html=True)

    #  DATA DRIFT 
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data Drift Check")

    recent = activity_df.head(50)["workload_label"].value_counts()
    old = activity_df.tail(50)["workload_label"].value_counts()

    c1, c2 = st.columns(2)
    c1.write("Recent Data")
    c1.bar_chart(recent)

    c2.write("Old Data")
    c2.bar_chart(old)

    st.markdown('</div>', unsafe_allow_html=True)
#  
# PREDICTION
with tab3:

    st.subheader("Select Model")

    model_choice = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Decision Tree"]
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

# Model path
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


    # INPUT
    c1, c2, c3, c4 = st.columns(4)

    assignments = c1.number_input("Assignments", 0, 10, 2)
    quizzes = c2.number_input("Quizzes", 0, 5, 1)
    exam = c3.selectbox("Exam Nearby", [0,1], format_func=lambda x: "Yes" if x else "No")
    prev = c4.selectbox("Previous Workload", [1,2,3],
                        format_func=lambda x: {1:"Low",2:"Medium",3:"High"}[x])

    # Predict button
    if st.button("Predict"):

# Create input
        inp = pd.DataFrame([{
            "assignments_count": assignments,
            "quizzes_count": quizzes,
            "exam_proximity": exam,
            "previous_workload": prev
        }])

        pred = model.predict(inp)[0]
        prob = max(model.predict_proba(inp)[0]) * 100

# Color based on result
        color = "#ff4d4d" if pred=="High" else "#ffc107" if pred=="Medium" else "#4caf50"


        # Show result
        st.markdown(f"""
        <div style="background:{card};padding:20px;border-left:6px solid {color}">
        <b>Model:</b> {model_choice} <br>
        <b>Prediction:</b> {pred} <br>
        <b>Confidence:</b> {prob:.1f}%
        </div>
        """, unsafe_allow_html=True)

        st.progress(prob/100)

        # SAVE prediction
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO weekly_predictions (week_no, predicted_workload) VALUES (%s, %s)",
            (int(activity_df["week_no"].max()) + 1, str(pred))
        )
        conn.commit()
        conn.close()
