# Part A – Data Preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1 line to load CSV
data = pd.read_csv(r"C:\Users\jayav\Downloads\Telegram Desktop\gameGuess\student_career_performance.csv")

# Explore
print(data.head())
print(data.info())
print(data.describe())

# Handle missing values by dropping (simple for beginners)
data = data.dropna()

# Drop duplicates
data = data.drop_duplicates()

# Features and targets
X = data[['Hours_Study', 'Sleep_Hours', 'Internships', 'Projects', 'CGPA']]
y_score = data['Placement_Score']
y_placed = data['Placed']

# Part B – Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)

print("Linear Regression Metrics:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Graph: predicted vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Placement Score")
plt.ylabel("Predicted Placement Score")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

# Part C – Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y_placed, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_class = log_model.predict(X_test)

print("Logistic Regression Metrics:")
print(classification_report(y_test, y_pred_class))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
y_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Part D – Insights (just printed)
print("Insights:")
print("1. Higher CGPA and more projects/internships generally correlate with higher placement scores.")
print("2. Students with fewer study hours but good CGPA might still get placed.")
print("3. Linear regression predicts scores roughly, logistic regression helps classify placement likelihood.")
