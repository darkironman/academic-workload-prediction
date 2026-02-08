# train_model.py
# This script trains the academic workload prediction model
# using data stored in MySQL and saves the trained model
# along with its performance metrics

import mysql.connector
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import os
from datetime import datetime
# Connect to the academic workload database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yashironman1@",
    database="academic_workload_db"
)
# Fetch academic activity data required for training
query = """
SELECT
    assignments_count,
    quizzes_count,
    exam_proximity,
    previous_workload,
    workload_label
FROM academic_activity
"""

df = pd.read_sql(query, conn)
conn.close()

# Remove records where workload label is missing
df = df.dropna(subset=["workload_label"])

# Separate input features and target variable
X = df[
    [
        "assignments_count",
        "quizzes_count",
        "exam_proximity",
        "previous_workload"
    ]
]

y = df["workload_label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression classification model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model performance on test data
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nModel Performance")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")

# Save trained model with timestamp for version control
os.makedirs("models", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_path = f"models/workload_model_{timestamp}.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved as: {model_path}")

# Store model performance details in database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yashironman1@",
    database="academic_workload_db"
)

cursor = conn.cursor()

cursor.execute(
    """
    INSERT INTO model_metadata (model_version, accuracy, precision_score, recall_score)
    VALUES (%s, %s, %s, %s)
    """,
    (
        timestamp,
        round(accuracy, 2),
        round(precision, 2),
        round(recall, 2),
    ),
)

conn.commit()
conn.close()

print("Model metadata stored in database")
print("Training completed successfully ")

