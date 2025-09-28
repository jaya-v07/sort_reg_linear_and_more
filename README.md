# sort_reg_linear_and_more
Student Placement Analysis
Overview

This project analyzes a dataset of students’ academic and extracurricular activities to predict:

Placement Score – How well a student might perform in placement tests/interviews.

Placement Status – Whether a student is likely to get placed (Yes/No).

We use Linear Regression for predicting scores and Logistic Regression for predicting placement status.

Dataset

The dataset contains the following columns:

Student_ID – Unique ID of the student

Hours_Study – Average study hours per day

Sleep_Hours – Average sleep per night

Internships – Number of internships completed

Projects – Number of academic/personal projects done

CGPA – Cumulative GPA (0–10)

Placement_Score – Composite score (0–100)

Placed – 1 = Yes, 0 = No

Steps

Load the dataset – Read CSV file in one line using pandas.

Data Cleaning – Drop missing values and duplicates for simplicity.

Feature Selection – Use study hours, sleep, internships, projects, and CGPA as input features.

Linear Regression – Predict Placement_Score and visualize predicted vs actual values.

Logistic Regression – Predict Placed (Yes/No), evaluate using classification metrics, confusion matrix, and ROC curve.

Insights – Simple takeaways about what factors affect placement readiness.

How to Run

Make sure Python is installed.

Install the required packages:

pip install pandas scikit-learn matplotlib seaborn


Place your CSV file in the same folder as the code.

Run the Python script.

Insights

Students with higher CGPA and more projects/internships generally have higher placement scores.

Even students with fewer study hours can get placed if they maintain a good CGPA.

Linear regression helps predict scores, while logistic regression helps classify placement chances.
