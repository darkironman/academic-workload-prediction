# add_weekly_data.py
# This script simulates live academic activity data
# by automatically inserting weekly records into the database

import mysql.connector
import random
import time
from datetime import datetime

print(" Live Academic Data Generator Started (Every 1 Minute)")

# Run continuously to simulate live data generation
while True:
    try:
          # Connect to MySQL database
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="YOUR_PASSWORD",
            database="academic_workload_db"
        )
        cursor = conn.cursor() 
       # Get the next week number automatically
        cursor.execute("SELECT MAX(week_no) FROM academic_activity")
        last_week = cursor.fetchone()[0]
        week_no = 1 if last_week is None else last_week + 1
    # Generate random academic activity data
        assignments = random.randint(1, 5)
        quizzes = random.randint(0, 2)
        exam_proximity = random.choice([0, 1])
        previous_workload = random.choice([1, 2, 3])
     
     # Assign workload label based on simple rules
        if assignments + quizzes >= 5 or exam_proximity == 1:
            workload = "High"
        elif assignments + quizzes >= 3:
            workload = "Medium"
        else:
            workload = "Low"

        # Insert generated data into academic_activity table
        cursor.execute("""
        INSERT INTO academic_activity
        (week_no, assignments_count, quizzes_count, exam_proximity, previous_workload, workload_label)
        VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            week_no,
            assignments,
            quizzes,
            exam_proximity,
            previous_workload,
            workload
        ))

        conn.commit()
        conn.close()

        print(f" [{datetime.now()}] Live data added for Week {week_no}")
  # Wait for 1 minute before inserting next record
        time.sleep(60)

    except Exception as e:
        # Handle database or runtime errors safely
        print(" Error:", e)
        time.sleep(60)

