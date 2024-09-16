# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :

1 . Import the required packages and print the present data.

2 . Print the placement data and salary data.

3 . Find the null and duplicate values.

4 . Using logistic regression find the predicted values of accuracy , confusion matrices.

5 . Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHIV SUJAN S R
RegisterNumber:  212223040194
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
data = pd.read_csv("/content/Placement_Data.csv")
data1 = data.drop(["sl_no", "salary"], axis=1)
le = LabelEncoder()
for col in data1.select_dtypes(include=['object']):
  data1[col] = le.fit_transform(data1[col])
x = data1.iloc[:, :-1]
y = data1["status"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[True, False])
cm_display.plot()
```
## Output:

![Screenshot 2024-09-14 103807](https://github.com/user-attachments/assets/af455455-8c8d-4ab1-8916-3911bad5c2e8)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
