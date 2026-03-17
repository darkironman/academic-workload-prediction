import mysql.connector              
import pandas as pd                
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, precision_score, recall_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier     
from sklearn.pipeline import Pipeline                
from sklearn.preprocessing import StandardScaler     
import pickle                                        
import os                                            
from datetime import datetime                        

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yashironman1@",
    database="academic_workload_db"
)

df = pd.read_sql("""
SELECT assignments_count, quizzes_count, exam_proximity, previous_workload, workload_label
FROM academic_activity
""", conn)

conn.close()

# PREPROCESSING 
df = df.dropna()
df = df.drop_duplicates()
df = df.sample(frac=1, random_state=42)

# FEATURES
X = df[["assignments_count","quizzes_count","exam_proximity","previous_workload"]]
y = df["workload_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LOGISTIC MODEL
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=200,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

# DECISION TREE
tree_model = DecisionTreeClassifier(max_depth=5)
tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)

#  PRINT RESULTS
print("\nModel Comparison")
print(f"Logistic Accuracy: {accuracy:.2f}")
print(f"Decision Tree Accuracy: {tree_accuracy:.2f}")

# SAVE MODEL
os.makedirs("models", exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_path = f"models/workload_model_{timestamp}.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)   

print(f"\nModel saved as: {model_path}")

# SAVE METADATA 
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yashironman1@",
    database="academic_workload_db"
)

cursor = conn.cursor()

cursor.execute("""
INSERT INTO model_metadata (model_version, accuracy, precision_score, recall_score)
VALUES (%s, %s, %s, %s)
""", (
    timestamp,
    float(accuracy),
    float(precision),
    float(recall)
))

conn.commit()
conn.close()

print("Metadata stored successfully")
print("Training completed successfully")
