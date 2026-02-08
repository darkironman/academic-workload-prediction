import mysql.connector
import pickle

with open("models/workload_model_v1.pkl", "rb") as f:
    model = pickle.load(f)

# Take user inputs for next week workload prediction
print("\nEnter details for NEXT WEEK workload prediction\n")

assignments = int(input("Number of assignments: "))
quizzes = int(input("Number of quizzes: "))
exam_proximity = int(input("Exam nearby? (1 = Yes, 0 = No): "))
previous_workload = int(input("Previous workload (1=Low, 2=Medium, 3=High): "))
week_no = int(input("Week number: "))

# Prepare input data in the format required by the model
X = [[assignments, quizzes, exam_proximity, previous_workload]]
# Predict workload level using the trained model
prediction = model.predict(X)[0]

# Convert numeric prediction into readable workload label
if prediction == 1:
    workload = "Low"
elif prediction == 2:
    workload = "Medium"
else:
    workload = "High"

print("\n Predicted Workload Level:", workload)
# Store predicted workload in the database for future reference
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yashironman1@",
    database="academic_workload_db"
)

cursor = conn.cursor()

cursor.execute("""
INSERT INTO weekly_predictions (week_no, predicted_workload)
VALUES (%s, %s)
""", (week_no, workload))

conn.commit()
conn.close()

print(" Prediction saved to database")
