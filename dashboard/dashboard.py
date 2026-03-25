import streamlit as st                      # main framework for building the web app UI
import mysql.connector                      # connects Python to MySQL database
import pandas as pd                         # handles data in table (DataFrame) format
import pickle                               # loads saved ML model from .pkl file
import os                                   # used to list files in the models folder

# set the browser tab title, layout width, and page icon
st.set_page_config(page_title="Workload Monitor", layout="wide", page_icon="📚")

# inject custom CSS styles to make the app look better
st.markdown("""
<style>
/* import Google font DM Sans for clean modern text */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');

/* apply DM Sans font to the entire app */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* make tab buttons fill full width of their column */
div[data-testid="column"] .tab-btn button {
    width: 100%;
}

/* style for the currently active tab — dark background with blue underline */
.tab-active button {
    background: #161B22 !important;
    border-bottom: 3px solid #58A6FF !important;
    color: #58A6FF !important;
    font-weight: 700 !important;
}

/* card used for showing a single metric (number + label) */
.metric-card {
    background: #161B22;           /* dark card background */
    border: 1px solid #21262D;     /* subtle border */
    border-radius: 12px;           /* rounded corners */
    padding: 22px 20px;            /* inner spacing */
    text-align: center;            /* centre all text */
    margin-bottom: 8px;            /* gap below each card */
}
.metric-card .val  { font-size: 2.2rem; font-weight: 700; color: #58A6FF; margin: 0; }  /* big blue number */
.metric-card .name { font-size: 0.95rem; font-weight: 600; color: #C9D1D9; }            /* metric name */
.metric-card .lbl  { font-size: 0.82rem; color: #8B949E; margin-top: 4px; }             /* small description */

/* card used for info/explanation blocks with a blue left border */
.info-card {
    background: #161B22;
    border-left: 4px solid #58A6FF; /* blue accent line on the left */
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: #C9D1D9;
    line-height: 1.75;              /* comfortable reading spacing */
}
.info-card b { color: #E6EDF3; }   /* bold text inside info cards is brighter white */

/* box that shows the prediction result */
.result-box {
    background: #161B22;
    border-radius: 14px;
    padding: 26px 28px;
    margin-top: 18px;
    line-height: 2;                 /* double line height for readability */
}
.result-box .level { font-size: 2rem; font-weight: 700; }          /* big label e.g. "High Workload" */
.result-box .conf  { font-size: 0.9rem; color: #8B949E; }          /* small confidence + model text */
.result-box .msg   { font-size: 1rem; color: #C9D1D9; }            /* advice message */

/* small uppercase section heading (e.g. "Step 1 — Choose a Model") */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #58A6FF;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)   # allow_unsafe lets HTML/CSS be injected directly

# ── Session state defaults ────────────────────────────────────

# initialise session state "tab" to "Overview" if it doesn't exist yet
# session_state keeps values alive across Streamlit reruns
if "tab" not in st.session_state:
    st.session_state.tab = "Overview"   # default tab shown when app first loads

# ── DB ────────────────────────────────────────────────────────

# cache this function — data is re-fetched only once every 60 seconds
@st.cache_data(ttl=60)
def load_data():
    # open connection to the local MySQL database
    conn = mysql.connector.connect(
        host="localhost",                    # database is running on this machine
        user="root",                         # MySQL username
        password="yashironman1@",            # MySQL password
        database="academic_workload_db"      # the specific database to use
    )
    # load all activity rows ordered by week number (oldest first)
    act  = pd.read_sql("SELECT * FROM academic_activity ORDER BY week_no ASC", conn)
    # load model training history ordered newest first
    meta = pd.read_sql("SELECT * FROM model_metadata ORDER BY trained_on DESC", conn)
    conn.close()       # always close the connection after reading
    return act, meta   # return both DataFrames

def save_prediction(week_no, label):
    # open a fresh connection just for writing the prediction
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="yashironman1@",
        database="academic_workload_db"
    )
    # insert the predicted workload for the given week into the predictions table
    conn.cursor().execute(
        "INSERT INTO weekly_predictions (week_no, predicted_workload) VALUES (%s, %s)",
        (week_no, label)   # %s placeholders prevent SQL injection
    )
    conn.commit()    # write the insert to disk
    conn.close()     # close connection

# ── Model Loader ──────────────────────────────────────────────

def load_model(choice: str):
    # always use logistic model
    keyword = "logistic"

    files = [f"models/{f}" for f in os.listdir("models") if f.endswith(".pkl")]

    if not files:
        return None

    # find only logistic model files
    matched = [f for f in files if keyword in f.lower()]

    # pick latest logistic model
    target = matched[0] if matched else max(files, key=os.path.getmtime)

    with open(target, "rb") as f:
        obj = pickle.load(f)

    # handle dict models
    if isinstance(obj, dict):
        for key in ["model", "classifier", "logistic"]:
            if key in obj:
                return obj[key]

        for v in obj.values():
            if hasattr(v, "predict"):
                return v

        return None

    return obj

# map numeric labels (1/2/3) to text labels ("Low"/"Medium"/"High")
# handles both int and string versions of 1, 2, 3 just in case
LABEL_MAP = {1:"Low", 2:"Medium", 3:"High", "1":"Low", "2":"Medium", "3":"High"}

def to_label(raw):
    # convert the raw model output to a clean text label
    label = LABEL_MAP.get(raw, raw)
    # if the result is already a valid label return it, otherwise default to "Low"
    return label if label in ("Low", "Medium", "High") else "Low"

#  Load data

# fetch activity records and model metadata from the database
activity_df, metadata_df = load_data()

#  Header

# display the main app title
st.markdown("##  Academic Workload Monitor")
# display a small subtitle below the title
st.caption("Track · Analyze · Predict your weekly academic load")
# draw a horizontal dividing line
st.divider()
 
# create 4 columns — first 3 hold tab buttons, last one is empty spacing
t1, t2, t3, _ = st.columns([1, 1, 1, 5])

# Overview tab button — clicking it sets the active tab in session state
with t1:
    if st.button(" Overview", use_container_width=True):
        st.session_state.tab = "Overview"

# Analysis tab button
with t2:
    if st.button("  Analysis", use_container_width=True):
        st.session_state.tab = "Analysis"

# Predict tab button
with t3:
    if st.button("  Predict", use_container_width=True):
        st.session_state.tab = "Predict"

# draw a separator line below the tab buttons
st.markdown("---")

# read the currently active tab from session state
tab = st.session_state.tab


# OVERVIEW TAB
if tab == "Overview":

    # show three info cards explaining what the app does
    st.markdown("""
    <div class="info-card"><b> What is this?</b><br>
    Tracks and predicts weekly academic workload using ML.
    Input your weekly data → get a <b>Low / Medium / High</b> forecast.</div>
    <div class="info-card"><b> How it works</b><br>
     Enter: assignments, quizzes, exam proximity, previous workload<br>
     Two ML models analyze patterns (Logistic Regression & Decision Tree)<br>
     Pick the model and get a prediction with confidence score</div>
    <div class="info-card"><b> Why it matters</b><br>
    Early warning so you can plan study time, reduce stress, and avoid cramming.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    # section heading for the data snapshot
    st.markdown("####  Dataset Snapshot")

    # create 4 equal columns for the 4 summary metrics
    c1, c2, c3, c4 = st.columns(4)

    # loop over each column and its corresponding metric data
    for col, (name, val, lbl) in zip([c1, c2, c3, c4], [
        ("Total Weeks",     activity_df["week_no"].nunique(),                   "weeks of data"),
        ("Avg Assignments", round(activity_df["assignments_count"].mean(), 1),  "per week"),
        ("Avg Quizzes",     round(activity_df["quizzes_count"].mean(), 1),      "per week"),
        ("Latest Week",     int(activity_df["week_no"].max()),                  "current week"),
    ]):
        # render each metric as a styled HTML card
        col.markdown(f"""<div class="metric-card">
            <p class="val">{val}</p><p class="name">{name}</p><p class="lbl">{lbl}</p>
        </div>""", unsafe_allow_html=True)

    # section heading for the data table
    st.markdown("####  Latest Records")
    # show the 10 most recent rows, newest first, without row index numbers
    st.dataframe(activity_df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)



# ANALYSIS TAB
elif tab == "Analysis":

    st.markdown("####  Weekly Workload Trend")
    st.caption("1 = Low · 2 = Medium · 3 = High")   # legend for the y-axis values

    # take the last 60 rows, convert text labels to numbers, set week_no as index for the chart
    trend = (activity_df.tail(60)
             .assign(Level=lambda d: d["workload_label"].map({"Low":1, "Medium":2, "High":3}))
             .set_index("week_no")[["Level"]])

    # draw the line chart in blue, 260px tall
    st.line_chart(trend, color=["#58A6FF"], height=260)
    st.divider()

    # split the next section into two side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("####  Workload Share")
        st.caption("How often each level appears")
        # count occurrences of each label, order Low→Medium→High, show as bar chart
        st.bar_chart(activity_df["workload_label"].value_counts()
                     .reindex(["Low", "Medium", "High"]).rename("Weeks"),
                     color=["#58A6FF"], height=240)

    with col2:
        st.markdown("####  Exam Impact")
        st.caption("Workload when exam is near vs not")
        # group rows by exam_proximity (0 or 1), count each workload label, pivot to wide format
        exam_df = (activity_df.groupby("exam_proximity")["workload_label"]
                   .value_counts().unstack().fillna(0)
                   .rename(index={0:"No Exam", 1:"Exam Near"})[["Low", "Medium", "High"]])
        st.bar_chart(exam_df, height=240)   # grouped bar chart comparing exam vs no exam

    st.divider()

    col3, col4 = st.columns(2)   # two more side-by-side charts

    with col3:
        st.markdown("####  Avg Assignments per Level")
        st.caption("More assignments = higher workload?")
        # average assignments_count grouped by workload label, ordered Low→Medium→High
        st.bar_chart(activity_df.groupby("workload_label")["assignments_count"]
                     .mean().reindex(["Low", "Medium", "High"]).rename("Avg"),
                     color=["#6EE7B7"], height=230)

    with col4:
        st.markdown("####  Avg Quizzes per Level")
        st.caption("More quizzes = higher workload?")
        # same idea but for quizzes_count
        st.bar_chart(activity_df.groupby("workload_label")["quizzes_count"]
                     .mean().reindex(["Low", "Medium", "High"]).rename("Avg"),
                     color=["#FCA5A5"], height=230)

    st.divider()

    st.markdown("####  Data Drift — Recent vs Older")
    st.caption("Are workload patterns changing over time?")

    c1, c2 = st.columns(2)   # side-by-side comparison of recent vs older data

    with c1:
        st.markdown("** Recent 50 Weeks**")
        # count label distribution in the most recent 50 rows
        st.bar_chart(activity_df.tail(50)["workload_label"]
                     .value_counts().reindex(["Low", "Medium", "High"]).rename("Count"),
                     color=["#58A6FF"], height=200)

    with c2:
        st.markdown("** Older 50 Weeks**")
        # count label distribution in the oldest 50 rows
        st.bar_chart(activity_df.head(50)["workload_label"]
                     .value_counts().reindex(["Low", "Medium", "High"]).rename("Count"),
                     color=["#F7A07E"], height=200)

    st.divider()

    st.markdown("####  Model Accuracy Over Time")

    if not metadata_df.empty:
        # plot accuracy values from training history, sorted by date, in purple
        st.line_chart(metadata_df.sort_values("trained_on")
                      .set_index("trained_on")[["accuracy"]]
                      .rename(columns={"accuracy": "Accuracy"}),
                      color=["#A78BFA"], height=220)
    else:
        st.info("No training history yet.")   # shown if no models have been trained

# PREDICT TAB
elif tab == "Predict":

    st.markdown("####  Predict Your Workload")

    # step 1 label
    st.markdown('<p class="section-label">Using Model: Logistic Regression</p>', unsafe_allow_html=True)

    # dropdown to select which ML model to use for prediction
    model_choice = "Logistic Regression"

    # load the selected model from the models folder
    model = load_model(model_choice)

    # stop the app if no model file was found
    if model is None:
        st.error(" No .pkl model files found in /models folder.")
        st.stop()   # halts execution — nothing below this runs

    # confirm which model was loaded successfully
    st.success(f" **{model_choice}** loaded", icon="✅")
    st.markdown("---")

    # step 2 label
    st.markdown('<p class="section-label">Step 2 — Enter Weekly Details</p>', unsafe_allow_html=True)

    # create 4 input columns for the 4 features
    c1, c2, c3, c4 = st.columns(4)

    # number input for how many assignments this week (0–10, default 2)
    assignments = c1.number_input(" Assignments", 0, 10, 2, key="asgn")

    # number input for how many quizzes this week (0–5, default 1)
    quizzes     = c2.number_input(" Quizzes",     0,  5, 1, key="quiz")

    # dropdown: is there an exam coming up? 0 = No, 1 = Yes
    exam        = c3.selectbox(" Exam Nearby", [0, 1], key="exam",
                               format_func=lambda x: " Yes" if x else " No")

    # dropdown: what was last week's workload? 1=Low, 2=Medium, 3=High
    prev        = c4.selectbox("⏮ Previous Load", [1, 2, 3], key="prev",
                               format_func=lambda x: {1:"🟢 Low", 2:"🟡 Medium", 3:"🔴 High"}[x])

    st.markdown("---")
    # step 3 label
    st.markdown('<p class="section-label">Step 3 — Get Prediction</p>', unsafe_allow_html=True)

    # big button that triggers the prediction when clicked
    if st.button(" Predict Workload", use_container_width=True, key="predict_btn"):

        # build a single-row DataFrame with the user's inputs matching the training feature names
        inp = pd.DataFrame([{
            "assignments_count": assignments,
            "quizzes_count":     quizzes,
            "exam_proximity":    exam,
            "previous_workload": prev
        }])

        # run the model and convert raw output to a clean label ("Low"/"Medium"/"High")
        st.session_state.pred_label = to_label(model.predict(inp)[0])

        # get the highest probability from predict_proba as a confidence percentage
        st.session_state.pred_conf  = round(max(model.predict_proba(inp)[0]) * 100, 1)

        # remember which model was used for this prediction
        st.session_state.pred_model = model_choice

        # try to save the prediction to the database — warn if it fails, don't crash
        try:
            save_prediction(int(activity_df["week_no"].max()) + 1, st.session_state.pred_label)
        except Exception as e:
            st.warning(f"Could not save: {e}")

    # show result block if a prediction exists in session state
    # this persists the result even when the user interacts with other widgets
    if "pred_label" in st.session_state:
        label = st.session_state.pred_label   # e.g. "High"
        conf  = st.session_state.pred_conf    # e.g. 87.3
        used  = st.session_state.pred_model   # e.g. "Logistic Regression"

        # colour for the result border and label text
        COLOR = {"High":"#FF6B6B", "Medium":"#FFC857", "Low":"#6EE7B7"}
        # emoji icon matching each label
        ICON  = {"High":"🔴",      "Medium":"🟡",       "Low":"🟢"}
        # advice message shown below the label
        MSG   = {
            "High":   "⚠️ Heavy week — prioritize tasks and manage time carefully.",
            "Medium": "🙂 Moderate week — stay consistent, don't fall behind.",
            "Low":    "✅ Light week — great time to review or get ahead.",
        }

        # render the result as a styled HTML box with a coloured left border
        st.markdown(f"""
        <div class="result-box" style="border-left: 6px solid {COLOR[label]};">
            <span class="level" style="color:{COLOR[label]};">{ICON[label]} {label} Workload</span><br>
            <span class="conf">Model: {used} &nbsp;|&nbsp; Confidence: {conf}%</span><br>
            <span class="msg">{MSG[label]}</span>
        </div>
        """, unsafe_allow_html=True)

        # show a progress bar representing confidence (0.0 to 1.0)
        st.progress(conf / 100)
        st.caption(" Prediction saved.")   # confirm the prediction was stored

    st.divider()

    # model performance metrics section
    st.markdown("####  Model Performance")

    if not metadata_df.empty:
        # read the latest model's scores from the first row (sorted newest first)
        acc  = metadata_df["accuracy"].iloc[0]
        prec = metadata_df["precision_score"].iloc[0]
        rec  = metadata_df["recall_score"].iloc[0]

        # calculate F1 score manually — harmonic mean of precision and recall
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

        # 4 columns for the 4 metric cards
        c1, c2, c3, c4 = st.columns(4)

        # render each metric as a styled card showing percentage
        for col, (name, val, desc) in zip([c1, c2, c3, c4], [
            ("Accuracy",  acc,  "overall correct"),
            ("Precision", prec, "correct positives"),
            ("Recall",    rec,  "coverage rate"),
            ("F1 Score",  f1,   "balanced score"),
        ]):
            col.markdown(f"""<div class="metric-card">
                <p class="val">{val*100:.1f}%</p>
                <p class="name">{name}</p><p class="lbl">{desc}</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No model metadata found.")   # shown if no training runs exist yet